#!/usr/bin/env python3

import logging
import time

from typing import List

import pyftdi.serialext

from serial import SerialBase
from pyftdi.ftdi import Ftdi
from pyftdi.usbtools import UsbDeviceDescriptor

from .configuration import BinOpt, Configuration

class ConfigurationError(Exception):
	pass

class FPGABoard:
	
	SCK = 1 # ADBUS0
	MOSI = 1 << 1 # ADBUS1
	MISO = 1 << 2 # ADBUS2
	# there is only one cs connected, but it is ADBUS4, not ADBUS3
	CS = 1 << 4 # ADBUS4
	CDONE = 1 << 6 # ADBUS6
	CRESET = 1 << 7 # ADBUS7
	
	def __init__(self, usb_dev: UsbDeviceDescriptor, baudrate: int=3000000, timeout: float=0.5) -> None:
		self._log = logging.getLogger(type(self).__name__)
		self._serial_number = usb_dev.sn
		self._is_open = False
		self._uart = pyftdi.serialext.serial_for_url(
			f"ftdi://::{usb_dev.bus}:{usb_dev.address}/2",
			baudrate=baudrate,
			timeout=timeout
		)
		self._direction = self.SCK|self.MOSI|self.CS|self.CRESET
		self._mpsse_dev = Ftdi()
		#self._mpsse_dev.log.setLevel(logging.DEBUG)
		self._mpsse_dev.open_mpsse_from_url(
			f"ftdi://::{usb_dev.bus}:{usb_dev.address}/1",
			direction=self._direction,
			initial=self._direction,
			frequency=6e6
		)
		self._is_open = True
	
	@property
	def uart(self) -> SerialBase:
		return self._uart
	
	@property
	def serial_number(self) -> str:
		return self._serial_number
	
	def close(self) -> None:
		self._close()
	
	def read_bytes(self, byte_count: int) -> bytes:
		response = self._uart.read(byte_count)
		# reverse from little endian
		return response[::-1]
	
	def read_integers(self, count: int=1, data_width: int=1) -> List[int]:
		res = []
		for _ in range(count):
			raw_data = self._uart.read(data_width)
			assert len(raw_data) == data_width, "Expected {} bytes, but got {}".format(data_width, len(raw_data))
			value = int.from_bytes(raw_data, 'little')
			res.append(value)
		
		return res
	
	def _close(self) -> None:
		if not self._is_open:
			return
		
		self._uart.close()
		self._mpsse_dev.close()
		self._is_open = False
	
	def reset_buffer(self, rst_input: bool=True, rst_output: bool=True) -> None:
		if rst_input:
			self._uart.reset_input_buffer()
		if rst_output:
			self._uart.reset_output_buffer()
	
	def flush(self) -> None:
		self._uart.flush()
	
	def __enter__(self) -> "FPGABoard":
		self._uart.reset_input_buffer()
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self._close()
		return False
	
	def configure(self, config: Configuration, opt: BinOpt=BinOpt()) -> None:
		bitstream = config.get_bitstream(opt)
		self.flash_bitstream(bitstream)
	
	def flash_bitstream(self, bitstream: bytes) -> None:
		self._flash_bitstream_spi(bitstream)
	
	def flash_bitstream_file(self, bitstream_path: str) -> None:
		self._flash_bitstream_file_spi(bitstream_path)
	
	def _flash_bitstream_file_spi(self, bitstream_path: str) -> None:
		# read bitstream
		with open(bitstream_path, "rb") as bitstream_file:
			bitstream = bitstream_file.read()
		
		self._flash_bitstream_spi(bitstream)
	
	def _flash_bitstream_spi(self, bitstream: bytes) -> None:
		self._log.debug("CDONE: %s", "high" if self._get_cdone() else "low")
		
		# creset to low
		# chip select to low to trigger configuration as SPI peripheral
		self._set_gpio_out(self.SCK|self.MOSI)
		
		self.usleep(100)
		
		self._set_gpio_out(self.SCK|self.MOSI|self.CRESET)
		
		# wait for FPGA to clear it's internal configuration memory
		# at least 1200 us
		self.usleep(1200)
		
		# construct flashing as single command
		cmd = bytearray((
			Ftdi.SET_BITS_LOW, self.MOSI|self.CS|self.CRESET, self._direction, # chip select high
			Ftdi.CLK_BITS_NO_DATA, 0x07, # 8 dummy clocks
			Ftdi.SET_BITS_LOW, self.MOSI|self.CRESET, self._direction # chip select low
		))
		
		# send bitstream
		chunk_size = 4096
		for i in range(0, len(bitstream), chunk_size):
			data = bitstream[i:i+chunk_size]
			cmd.append(Ftdi.WRITE_BYTES_NVE_MSB)
			len_data = len(data) - 1
			cmd.append(len_data & 0xff)
			cmd.append(len_data >> 8)
			cmd.extend(data)
		
		
		# chip select to high
		self._extend_by_gpio(cmd, self.MOSI|self.CS|self.CRESET)
		
		# wait 100 SPI clock cycles for CDONE to go high
		cmd.extend((Ftdi.CLK_BYTES_NO_DATA, 0x0b, 0x00, Ftdi.CLK_BITS_NO_DATA, 0x03))
		
		self._log.debug("Write flash command")
		self._mpsse_dev.write_data(cmd)
		
		self._log.debug("Check success of flash")
		# check CDONE
		if self._get_cdone():
			self._log.debug("CDONE: high, programming successful")
		else:
			raise ConfigurationError("Programming failed")
		
		# wait at least 49 SPI clock cycles
		cmd = bytearray((Ftdi.CLK_BYTES_NO_DATA, 0x05, 0x00, Ftdi.CLK_BITS_NO_DATA, 0x00))
		self._extend_by_gpio(cmd, self._direction)
		self._mpsse_dev.write_data(cmd)
		
		# SPI pins now also available as user IO (from the FPGA perspective), but they are not used
	
	def _set_gpio_out(self, value: int) -> None:
		cmd = bytearray()
		self._extend_by_gpio(cmd, value)
		self._mpsse_dev.write_data(cmd)
	
	def _extend_by_gpio(self, cmd: bytearray, value: int) -> None:
		cmd.extend((Ftdi.SET_BITS_LOW, value, self._direction))
	
	def _get_cdone(self) -> bool:
		cmd = bytes((Ftdi.GET_BITS_LOW, Ftdi.SEND_IMMEDIATE))
		self._mpsse_dev.write_data(cmd)
		gpio = self._mpsse_dev.read_data_bytes(1, 4)[0]
		return (gpio & self.CDONE) != 0
	
	@classmethod
	def get_suitable_board(cls, baudrate: int=3000000, timeout: int=0.5) -> "FPGABoard":
		suitable = set()
		ft2232_devices = Ftdi.find_all([(0x0403, 0x6010)], True)
		
		for desc, i_count in ft2232_devices:
			if i_count==2:
				suitable.add(desc)
		
		if len(suitable) == 0:
			raise Exception("No suitable devices found.")
		
		return cls(suitable.pop(), baudrate, timeout)
	
	@staticmethod
	def usleep(usec: float) -> None:
		time.sleep(usec/1000000)
