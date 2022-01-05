#!/usr/bin/env python3

import binascii

from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import BinaryIO, List, NewType, Optional

import numpy as np

from .device_data import DeviceSpec, TilePosition, TileType, SPECS_BY_ASC

Banks = NewType("Banks", np.ndarray)

class FreqRange(IntEnum):
	"""Values for internal oscillator frequncy range
	
	Relevant for configuration in SPI master mode.
	Depends on thr PROM speed.
	"""
	LOW = 0
	MEDIUM = 1
	HIGH = 2

class MalformedBitstreamError(Exception):
	"""Raised when an incorrect bitstream is encountered."""

@dataclass
class BinOpt:
	"""Options for binary bitstream creation"""
	bram_chunk_size: int = 128
	# detect used BRAM tiles and write the BRAM bank accordingly, if False bram_banks will be used
	detect_used_bram: bool = False
	# banks to write, None means write all (!), empty List means skip BRAM, used if detect_used_bram is False
	bram_banks: Optional[List[int]] = None
	# 0 -> no optimization, write all CRAM, one block per bank
	# 1 -> set bank width, height, number and offset only if differs from currently set value
	# 2 -> in addition to 1, skip CRAM banks that only contain zeros
	# 3 -> in addition to 2, skip CRAM lines that only contain zeros,
	#      i.e. write only chunks where each line contains at least one one
	# 4 -> in addition to 3, order writes by chunk size to reduce number of set height commands
	optimize: int = 0
	skip_comment: bool = False

class CRC:
	def __init__(self) -> None:
		self.reset()
	
	@property
	def value(self) -> int:
		return self._value
	
	def reset(self) -> None:
		self._value = 0xFFFF
	
	def update(self, data: bytes) -> None:
		self._value = binascii.crc_hqx(data, self._value)

NOSLEEP_MASK = 1
WARMBOOT_MASK = 1<<5

class BinOut:
	"""Wrapper around BinaryIO to provide functions for creating binary bitstreams"""
	
	BOOLS_TO_BYTES = {tuple(v&1<<(7-i) != 0 for i in range(8)): v for v in range(256)}
	
	def __init__(self, bin_file: BinaryIO) -> None:
		self._bin_file = bin_file
		self._crc = CRC()
		self._bank_number = None
		self._bank_width = None
		self._bank_height = None
		self._bank_offset = None
		
	
	@property
	def bank_height(self) -> int:
		return self._bank_height
	
	def write_bytes(self, data: bytes) -> None:
		"""Writes bytes, updates CRC accordingly"""
		count = self._bin_file.write(data)
		
		if count != len(data):
			raise IOError(f"only {count} of {len(data)} bytes written")
		
		self._crc.update(data)
	
	def write_comment(self, comment: str) -> None:
		if not comment:
			return
		
		self.write_bytes(b"\xff\x00")
		
		if comment[-1] == "\n":
			comment = comment[:-1]
		for line in comment.split("\n"):
			self.write_bytes(line.encode("utf-8"))
			self.write_bytes(b"\x00")
		
		self.write_bytes(b"\x00\xff")
	
	def write_preamble(self) -> None:
		self.write_bytes(b"\x7e\xaa\x99\x7e")
	
	def write_freq_range(self, freq_range: FreqRange) -> None:
		self.write_bytes(b"\x51")
		self.write_bytes(bytes([int(freq_range)]))
	
	def crc_reset(self) -> None:
		self.write_bytes(b"\x01\x05")
		self._crc.reset()
	
	def write_warmboot(self, warmboot: bool, nosleep: bool) -> None:
		"""Writes warmboot and nosleep flags"""
		self.write_bytes(b"\x92\x00")
		wn = 0
		if nosleep:
			wn |= NOSLEEP_MASK
		if warmboot:
			wn |= WARMBOOT_MASK
		self.write_bytes(bytes([wn]))
	
	def set_bank_number(self, number: int, reuse: bool=False) -> None:
		if reuse and self._bank_number == number:
			return
		
		self.write_bytes(b"\x11")
		self.write_bytes(bytes([number]))
		self._bank_number = number
	
	def set_bank_width(self, width: int, reuse: bool=False) -> None:
		if reuse and self._bank_width == width:
			return
		
		self.write_bytes(b"\x62")
		self.write_bytes((width-1).to_bytes(2, "big"))
		self._bank_width = width
	
	def set_bank_height(self, height: int, reuse: bool=False) -> None:
		if reuse and self._bank_height == height:
			return
		
		self.write_bytes(b"\x72")
		self.write_bytes(height.to_bytes(2, "big"))
		self._bank_height = height
	
	def set_bank_offset(self, offset: int, reuse: bool=False) -> None:
		if reuse and self._bank_offset == offset:
			return
		
		self.write_bytes(b"\x82")
		self.write_bytes(offset.to_bytes(2, "big"))
		self._bank_offset = offset
	
	def data_from_xram(self, xram: Banks) -> bytes:
		data = np.packbits(xram[self._bank_number, self._bank_offset:self._bank_offset+self._bank_height, :self._bank_width], bitorder="big").tobytes()
		if len(data)*8 != self._bank_height*self._bank_width:
			print(f"padding: {self._bank_height*self._bank_width} to {len(data)*8} {data[-1]:08b}")
		return data
	
	def write_cram(self, cram: Banks) -> None:
		data = self.data_from_xram(cram)
		self.write_bytes(b"\x01\x01")
		self.write_bytes(data)
		self.write_bytes(b"\x00\x00")
	
	def write_bram(self, bram: Banks) -> None:
		data = self.data_from_xram(bram)
		self.write_bytes(b"\x01\x03")
		self.write_bytes(data)
		self.write_bytes(b"\x00\x00")
	
	def crc_check(self) -> None:
		self.write_bytes(b"\x22")
		self.write_bytes(self._crc.value.to_bytes(2, "big"))
	
	def wakeup(self) -> None:
		self.write_bytes(b"\x01\x06")
	

class Configuration:
	"""Represents the configuration of a FPGA"""
	
	def __init__(self, device_spec: DeviceSpec) -> None:
		self._spec = device_spec
		self.clear()
	
	def clear(self) -> None:
		self._bram = {}
		self._tiles = {}
		self._tiles_by_type = {}
		self._tile_types = {}
		self._comment = ""
		self._freq_range = FreqRange.LOW
		self._warmboot = True
		self._nosleep = False
		self._extra_bits = []
		
		for pos, ttype in self._spec.get_tile_types():
			width = self._spec.tile_type_width[ttype]
			height = self._spec.tile_height
			data = np.full((height, width), False, dtype=bool)
			self._tiles[pos] = data
			self._tiles_by_type.setdefault(ttype, []).append(pos)
			self._tile_types[pos] = ttype
			
			if ttype == TileType.RAM_B:
				self._bram[pos] = np.full((16, 256), False, dtype=bool)
	
	def get_bit(self, x: int, y: int, group: int, index: int) -> bool:
		return self._tiles[(x, y)][group, index]
	
	def _all_blank_cram_banks(self) -> Banks:
		"""Creates all CRAM banks as used in binary bitstreams with all bits set to 0.
		
		Attention: the access to the bank b at x, y is reached by banks[b][y][x] to easier group the bits in the
		x dimension.
		"""
		return np.full((4, self._spec.cram_height, self._spec.cram_width), False, dtype=bool)
	
	def _all_blank_bram_banks(self) -> Banks:
		"""Creates all BRAM banks as used in binary bitstreams with all bits set to 0.
		
		Attention: the access to the bank b at x, y is reached by banks[b][y][x] to easier group the bits in the
		x dimension.
		"""
		return np.full((4, self._spec.bram_height, self._spec.bram_width), False, dtype=bool)
	
	def read_bin(self, bin_file: BinaryIO):
		"""Reads binary bitstream"""
		crc = CRC()
		
		start_word = self.get_bytes_crc(bin_file, 2, crc)
		if start_word == b"\xff\x00":
			# read multiple null terminated comments
			com_list = []
			prv = b"\x00" # from 0xFF00
			cur = self.get_bytes_crc(bin_file, 1, crc)
			while True:
				while cur != b"\x00":
					com_list.append(cur)
					prv = cur
					cur = self.get_bytes_crc(bin_file, 1, crc)
				
				nxt = self.get_bytes_crc(bin_file, 1, crc)
				if nxt == b"\xff":
					if prv == b"\x00":
						# previous string null terminated and received 0x00FF
						# -> normal end of comment
						break
					else:
						# previous string not null terminated
						# -> Lattice bug that shifts 0x00FF some bytes into comments
						com_list.append(b"\n")
						break
				else:
					# another comment string
					prv = cur
					cur = nxt
				com_list.append(b"\n")
			
			self._comment = b"".join(com_list).decode("utf-8")
			last_four = [None]*4
		else:
			# no comment
			self._comment = ""
			last_four = [None, None, start_word[:1], start_word[1:]]
		
		# as Lattice' own tools create faulty comments just search for preamble instead of expecting it
		while last_four != [b"\x7e", b"\xaa", b"\x99", b"\x7e"]:
			last_four = last_four[1:]
			last_four.append(self.get_bytes_crc(bin_file, 1, crc))
		
		bank_nr = None
		bank_width = None
		bank_height = None
		bank_offset = None
		
		def get_data_len():
			try:
				return (bank_width*bank_height+7)//8 # round up
			except TypeError as te:
				raise MalformedBitstreamError("Block height and width have to be set before writig data") from te
		
		def data_to_xram(data, xram):
			for y in range(bank_height):
				fst = y*bank_width//8
				lst = ((y+1)*bank_width+7)//8 # round up
				# msb first
				bit_data = [
					(b<<i) & 0x80 != 0 for c, b in enumerate(data[fst:lst])
					for i in range(max(0, y*bank_width-8*fst-8*c), min(8, (y+1)*bank_width-8*fst-8*c))
				]
				xram[bank_nr][y+bank_offset][0:bank_width] = bit_data
			
		
		cram = self._all_blank_cram_banks()
		bram = self._all_blank_bram_banks()
		while True:
			file_offset = bin_file.tell()
			# don't use get_bytes as the end of the file should be detected here
			raw_com = bin_file.read(1)
			if len(raw_com) == 0:
				# end of file
				break
			crc.update(raw_com)
			
			command = raw_com[0]
			opcode = command >> 4
			payload_len = command & 0xf
			
			payload_bytes = self.get_bytes_crc(bin_file, payload_len, crc)
			payload = 0
			for val in payload_bytes:
				payload = payload << 8 | val
			
			if opcode == 0:
				if payload == 1:
					if bank_width != self._spec.cram_width:
						raise ValueError(f"Wrong CRAM width: expected {self._spec.cram_width},but was {bank_width}")
					data_len = get_data_len()
					data = self.get_bytes_crc(bin_file, data_len, crc)
					data_to_xram(data, cram)
					self.expect_bytes(bin_file, b"\x00\x00", crc, "Expected 0x{exp} after CRAM data, got 0x{val}")
				elif payload == 3:
					if bank_width != self._spec.bram_width:
						raise ValueError(f"Wrong BRAM width: expected {self._spec.bram_width},but was {bank_width}")
					data_len = get_data_len()
					data = self.get_bytes_crc(bin_file, data_len, crc)
					data_to_xram(data, bram)
					self.expect_bytes(bin_file, b"\x00\x00", crc, "Expected 0x{exp} after BRAM data, got 0x{val}")
				elif payload == 5:
					crc.reset()
				elif payload == 6:
					# wakeup -> ignore everything after that
					break
				else:
					# payload 8 (reboot) not supported
					raise MalformedBitstreamError(f"Unsupported Command: 0x{command:02x} 0x{payload:0{payload_len*2}x}")
			elif opcode == 1:
				bank_nr = payload
			elif opcode == 2:
				if crc.value != 0:
					raise MalformedBitstreamError(f"Wrong CRC is {crc.value:04x}")
			elif opcode == 5:
				try:
					self._freq_range = FreqRange(payload)
				except ValueError as ve:
					raise MalformedBitstreamError(f"Unknown value for frequency range {payload}") from ve
			elif opcode == 6:
				bank_width = payload + 1
			elif opcode == 7:
				bank_height = payload
			elif opcode == 8:
				bank_offset = payload
			elif opcode == 9:
				self._nosleep = (payload & NOSLEEP_MASK) != 0
				self._warmboot = (payload & WARMBOOT_MASK) != 0
			else:
				# opcode 4 (set boot address) not supported
				raise MalformedBitstreamError(f"Unknown opcode {opcode:1x} at 0x{file_offset}")
		
		self._read_cram_banks(cram)
		self._read_bram_banks(bram)
	
	def _get_cram_banks(self) -> Banks:
		cram = self._all_blank_cram_banks()
		self._write_cram_banks(cram)
		
		return cram
	
	def _get_bram_banks(self) -> Banks:
		bram = self._all_blank_bram_banks()
		self._write_bram_banks(bram)
		
		return bram
	
	def get_bitstream(self, opt: BinOpt=BinOpt()) -> bytes:
		"""Returns binary bitstream"""
		with BytesIO() as bin_file:
			self.write_bin(bin_file, opt)
			bitstream = bin_file.getvalue()
		
		return bitstream
	
	def write_bin(self, bin_file: BinaryIO, opt: BinOpt=BinOpt()):
		"""Writes binary bitstream"""
		cram = self._get_cram_banks()
		bram = self._get_bram_banks()
		
		bin_out = BinOut(bin_file)
		
		reuse = opt.optimize > 0
		
		# comment
		comment = self._comment
		if opt.skip_comment:
			comment = None
		bin_out.write_comment(comment)
		
		# preamble
		bin_out.write_preamble()
		
		# frequency range
		bin_out.write_freq_range(self._freq_range)
		
		# CRC reset
		bin_out.crc_reset()
		
		# warmboot & nosleep
		bin_out.write_warmboot(self._warmboot, self._nosleep)
		
		if opt.optimize < 2:
			out_banks = list(range(len(cram)))
		else:
			out_banks = [c for c in range(len(cram)) if np.any(cram[c])]
		
		if len(out_banks):
			# bank width
			bin_out.set_bank_width(self._spec.cram_width, reuse)
			# bank height
			bin_out.set_bank_height(self._spec.cram_height, reuse)
			
			if opt.optimize < 3:
				# bank offset
				bin_out.set_bank_offset(0, reuse)
		
		# write CRAM
		granularity = np.lcm(self._spec.cram_width, 8)//self._spec.cram_width
		for bank_number in out_banks:
			bin_out.set_bank_number(bank_number, reuse)
			
			if opt.optimize < 3:
				# write whole bank at once
				bin_out.write_cram(cram)
				continue
			
			areas = self.get_nonzero_chunks(cram[bank_number], granularity, False)
			
			if opt.optimize >= 4:
				# sort by chunk size to set bank height less often
				areas = areas[areas[:, 1].argsort()]
			
			for bank_offset, bank_height in areas:
				bin_out.set_bank_height(int(bank_height), True)
				bin_out.set_bank_offset(int(bank_offset), True)
				bin_out.write_cram(cram)
		
		# write BRAM
		if opt.detect_used_bram:
			out_banks = [b for b in range(len(bram)) if self._bram_bank_used(b)]
		elif opt.bram_banks is None:
			out_banks = list(range(len(bram)))
		else:
			out_banks = opt.bram_banks
		
		if len(out_banks):
			chunk_size = opt.bram_chunk_size
			bin_out.set_bank_width(self._spec.bram_width)
			for bank_number in out_banks:
				# may be set to value different from chunk_size by previous bank to write last, smaller chunk
				if bin_out.bank_height != chunk_size:
					bin_out.set_bank_height(chunk_size, reuse)
				bin_out.set_bank_number(bank_number)
				for bank_offset in range(0, self._spec.bram_height, chunk_size):
					if bank_offset + chunk_size > self._spec.bram_height:
						bin_out.set_bank_height(self._spec.bram_height - bank_offset, reuse)
					bin_out.set_bank_offset(bank_offset, reuse)
					bin_out.write_bram(bram)
		
		# CRC check
		bin_out.crc_check()
		
		# wakeup
		bin_out.wakeup()
		
		# padding
		bin_out.write_bytes(b"\x00")
	
	def _bram_bank_used(self, bank_number: int) -> bool:
		tile_x = self._spec.bram_cols[bank_number//2]
		y_offset = (self._spec.max_y-1)//2 if bank_number%2 == 1 else 0
		indi = self._spec.bram_indicator
		for block_nr in range(self._spec.bram_width//16):
			tile_y = block_nr*2 + 1 + y_offset
			if self.get_bit(tile_x+indi.offset.x, tile_y+indi.offset.y, *indi.bit) == indi.value:
				return True
		
		return False
	
	def _read_cram_banks(self, cram: Banks) -> None:
		self._access_cram_banks(cram, True)
	
	def _write_cram_banks(self, cram: Banks) -> None:
		self._access_cram_banks(cram, False)
	
	def _access_cram_banks(self, cram: Banks, read: bool) -> None:
		for bank_nr, cram_bank in enumerate(cram):
			top = bank_nr%2 == 1
			right = bank_nr >= 2
			
			if top:
				y_range = list(reversed(range((self._spec.max_y+1)//2, self._spec.max_y)))
				io_y = self._spec.max_y
			else:
				y_range = list(range(1, (self._spec.max_y+1)//2))
				io_y = 0
			
			if right:
				x_range = list(reversed(range((self._spec.max_x+1)//2, self._spec.max_x+1)))
			else:
				x_range = list(range((self._spec.max_x+1)//2))
			
			# IO in x direction
			x_off = self._spec.tile_type_width[self._tile_types[TilePosition(x_range[0], y_range[0])]]
			for tile_x in x_range[1:]:
				# width is defined by the other tile i the row, not the IO tile
				row_width = self._spec.tile_type_width[self._tile_types[TilePosition(tile_x, y_range[0])]]
				
				tile_data = self._tiles[TilePosition(tile_x, io_y)]
				
				cram_x_idx = [23, 25, 26, 27, 16, 17, 18, 19, 20, 14, 32, 33, 34, 35, 36, 37, 4, 5]
				if right:
					cram_x_idx = [row_width-1-i for i in cram_x_idx]
				
				cram_y_idx = np.array([15, 14, 12, 13, 11, 10, 8, 9, 7, 6, 4, 5, 3, 2, 0, 1])
				if read:
					tile_data[:, :] = cram_bank[cram_y_idx[:,np.newaxis],[x_off+i for i in cram_x_idx]]
				else:
					cram_bank[cram_y_idx[:,np.newaxis],[x_off+i for i in cram_x_idx]] = tile_data
				
				x_off += row_width
			
			y_off = self._spec.tile_height
			for tile_y in y_range:
				x_off = 0
				for tile_x in x_range:
					tile_pos = TilePosition(tile_x, tile_y)
					tile_data = self._tiles[tile_pos]
					tile_type = self._tile_types[tile_pos]
					tile_width = self._spec.tile_type_width[tile_type]
					
					cram_x_slice = slice(x_off, x_off+tile_width)
					if right or tile_type == TileType.IO:
						cram_x_slice = self.reverse_slice(cram_x_slice)
					index_slice = slice(0, tile_width)
					
					cram_y_slice = slice(y_off, y_off+self._spec.tile_height)
					if top:
						cram_y_slice = self.reverse_slice(cram_y_slice)
					
					if read:
						tile_data[:self._spec.tile_height, index_slice] = cram_bank[cram_y_slice, cram_x_slice]
					else:
						cram_bank[cram_y_slice, cram_x_slice] = tile_data[:self._spec.tile_height, index_slice]
					
					x_off += tile_width
				y_off += self._spec.tile_height
		
		# extra bits
		if read:
			self._extra_bits = []
			for extra in self._spec.extra_bits:
				if cram[extra.bank, extra.y, extra.x]:
					self._extra_bits.append(extra)
		else:
			for extra in self._extra_bits:
				cram[extra.bank, extra.y, extra.x] = True
	
	def _read_bram_banks(self, bram: Banks) -> None:
		self._access_bram_banks(bram, True)
	
	def _write_bram_banks(self, bram: Banks) -> None:
		self._access_bram_banks(bram, False)
	
	def _access_bram_banks(self, bram: Banks, read: bool) -> None:
		for bank_nr, bram_bank in enumerate(bram):
			top = bank_nr%2 == 1
			tile_x = self._spec.bram_cols[bank_nr//2]
			for block_nr in range(self._spec.bram_width//16):
				tile_y = block_nr*2 + 1
				if top:
					# in fact it should be (max_y-1)//2 but as max_y is always odd it yields the same result
					tile_y += self._spec.max_y//2
				bram_data = self._bram[TilePosition(tile_x, tile_y)]
				
				row_slice = slice((block_nr+1)*16-1, block_nr*16-1 if block_nr else None, -1)
				
				if read:
					bram_data[:, :] = np.reshape(bram_bank[:, row_slice], (16, 256))
				else:
					bram_bank[:, row_slice] = np.reshape(bram_data, (256, 16))
	
	@staticmethod
	def reverse_slice(org_slice: slice) -> slice:
		# only tested for |step| == 1
		step = -(org_slice.step or 1)
		if step < 0:
			# org step was positive
			
			if org_slice.stop == 0:
				# special case, always returns empty list
				return slice(org_slice.start, -1, step)
			
			if org_slice.start in (None, 0):
				stop = None
			else:
				stop = org_slice.start - 1
			
			if org_slice.stop is None:
				start = None
			else:
				start = org_slice.stop - 1
		else:
			# org step was negative
			
			if org_slice.stop == -1:
				# special case, always returns empty list
				return slice(org_slice.start, 0, step)
			
			if org_slice.start in (None, -1):
				stop = None
			else:
				stop = org_slice.start + 1
			
			if org_slice.stop is None:
				start = None
			else:
				start = org_slice.stop + 1
		
		return slice(start, stop, step)
	
	@staticmethod
	def set_with_granularity_aligned(to_write: np.ndarray, granularity: int) -> None:
		assert len(to_write) % granularity == 0
		
		for i in range(0, len(to_write), granularity):
			if any(to_write[i:i+granularity]):
				to_write[i:i+granularity] = [True]*granularity
	
	@staticmethod
	def set_with_granularity_unaligned(to_write: np.ndarray, granularity: int) -> None:
		i = 0
		while i < len(to_write):
			if not to_write[i]:
				i += 1
				continue
			for j in range(1, granularity):
				try:
					to_write[i+j] = True
				except IndexError:
					break
			i += j + 1
	
	@classmethod
	def get_nonzero_chunks(cls, data: np.ndarray, granularity: int, align: bool) -> np.ndarray:
		"""Extracts which rows of data are nonzero and group them in chunks.
		The length of each chunk will be divisable by the granularity. If the align flag is set the offset of each
		chunk will also be divisable by the granularity.
		
		Returns a 2d-array with a row for each chunk, the first column as the offset of the chunks (i.e. the index
		of the first data row of the chunks) the second column as the chunk length.
		"""
		# find none False rows
		to_write = np.any(data, axis=1)
		if granularity > 1:
			if align:
				cls.set_with_granularity_aligned(to_write, granularity)
			else:
				cls.set_with_granularity_unaligned(to_write, granularity)
			
		prev_write = np.roll(to_write, 1)
		prev_write[0] = False
		next_write = np.roll(to_write, -1)
		next_write[-1] = False
		first_indices = np.nonzero(to_write & ~prev_write)[0]
		last_indices = np.nonzero(to_write & ~next_write)[0]
		chunk_size = last_indices - first_indices + 1
		
		if not align and chunk_size[-1] % granularity != 0:
			# for not aligned chunks the last chunk can be too short as there may be not enough rows following the last
			# nonzero row to get a chunk of granularity size
			# -> simply add the rows before to the chunk
			# the added rows may be written twice
			cor = granularity - (chunk_size[-1] % granularity)
			chunk_size[-1] += cor
			first_indices[-1] -= cor
		assert all([c%granularity==0 for c in chunk_size]) # only full bytes are written to CRAM
		
		return np.column_stack((first_indices, chunk_size))
	
	@classmethod
	def expect_bytes(cls, bin_file: BinaryIO, exp: bytes, crc: CRC, msg: str="Expected {exp} but got {val}") -> None:
		"""Reads bytes and raise exception if they do not match the expected values"""
		val = cls.get_bytes_crc(bin_file, len(exp), crc)
		
		if exp != val:
			raise MalformedBitstreamError(msg.format(exp=exp, val=val))
	
	@classmethod
	def get_bytes_crc(cls, bin_file: BinaryIO, size: int, crc: CRC) -> bytes:
		"""Gets a specific number of bytes and update a CRC"""
		res = cls.get_bytes(bin_file, size)
		crc.update(res)
		return res
	
	@staticmethod
	def get_bytes(bin_file: BinaryIO, size: int) -> bytes:
		"""Get a specific number of bytes"""
		res = bin_file.read(size)
		
		if len(res) < size:
			raise EOFError()
		
		return res
	
	@classmethod
	def create_blank(cls, asc_name: str="8k") -> "Configuration":
		"""Creates an empty configuration"""
		spec = SPECS_BY_ASC[asc_name]
		config = cls(spec)
		
		return config
