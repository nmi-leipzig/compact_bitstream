#!/usr/bin/env python3

import statistics
import time
import timeit

from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
from io import BytesIO
from typing import Any, BinaryIO, Callable, List, Tuple

import h5py

from src.configuration import BinOpt, Configuration
from src.fpga_board import FPGABoard

def bitstream_from_args(in_stream: BinaryIO, args: Namespace) -> bytes:
	config = Configuration.create_blank(args.chip_type)
	
	config.read_bin(in_stream)
	
	opt = BinOpt(
		bram_chunk_size = args.bram_chunk_size,
		detect_used_bram = False,
		bram_banks = args.bram_banks,
		optimize = args.level,
		skip_comment = args.skip_comment,
	)
	
	return config.get_bitstream(opt)

def benchmark(in_stream: BinaryIO, args: Namespace) -> List[float]:
	bitstream = bitstream_from_args(in_stream, args)
	
	with FPGABoard.get_suitable_board() as fpga:
		res = timeit.repeat("fpga.flash_bitstream(bitstream)", repeat=args.repeat, number=args.number, globals={"fpga": fpga, "bitstream": bitstream})
	
	return res

def exec_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
	start = time.perf_counter()
	res = func(*args, **kwargs)
	end = time.perf_counter()
	
	return res, end-start

def run(args: Namespace) -> None:
	with ExitStack() as stack:
		in_batch = []
		if args.input_bin:
			bin_file = stack.enter_context(open(args.input_bin, "rb"))
			in_batch.append(bin_file)
		hdf5_file = None
		if args.hdf5:
			hdf5_file = stack.enter_context(h5py.File(args.hdf5, "a"))
		name_list = None
		if hdf5_file and args.input_path:
			in_grp = hdf5_file[args.input_path]
			if isinstance(in_grp, h5py.Dataset):
				ds_list = [in_grp]
			else:
				name_ds_list = [g for g in in_grp.items() if isinstance(g[1], h5py.Dataset)]
				name_list = [n[0] for n in name_ds_list]
				ds_list = [n[1] for n in name_ds_list]
			in_batch.extend([BytesIO(d[:].tobytes()) for d in ds_list])
		
		timing_res = [exec_time(args.function, b, args) for b in in_batch]
		res = [r[0] for r in timing_res]
		timing = [r[1] for r in timing_res]
		
		if hdf5_file and args.result_path:
			if args.function == benchmark:
				dtype = "float64"
			else:
				dtype = "uint8"
				res = [bytearray(r) for r in res]
			
			if name_list:
				try:
					out_grp = hdf5_file[args.result_path]
				except KeyError:
					out_grp = hdf5_file.create_group(args.result_path)
				for name, data, seconds in zip(name_list, res, timing):
					dataset = out_grp.create_dataset(name, data=data, dtype=dtype, compression="gzip", compression_opts=9)
					dataset.attrs.create("execution_time_in_s", data=seconds, dtype=float)
			else:
				out_grp = hdf5_file.create_dataset(args.result_path, data=res[0], dtype=dtype, compression="gzip", compression_opts=9)
				out_grp.attrs.create("execution_time_in_s", data=timing[0], dtype=float)
			
			# set args as out_grp attributes
			if args.description:
				out_grp.attrs["description"] = args.description
			out_grp.attrs.create("optimization_level", data=args.level, dtype="uint8")
			for attr_name, dtype in [("chip_type", str), ("bram_banks", "uint8"), ("skip_comment", bool), ("bram_chunk_size", "uint16")]:#
				value = getattr(args, attr_name)
				if value is None:
					out_grp.attrs[attr_name] = h5py.Empty(dtype)
					continue
				if dtype is str:
					out_grp.attrs[attr_name] = value
					continue
				out_grp.attrs.create(attr_name, data=value, dtype=dtype)
			
		elif args.function == benchmark:
			print(res)
			print(f"stdev: {statistics.stdev(res)}, mean: {statistics.mean(res)}")
		
		if args.function == bitstream_from_args and args.output_bin:
			with open(args.output_bin, "wb") as bin_file:
				bin_file.write(res[0])
			

def create_arg_parser() -> ArgumentParser:
	parser = ArgumentParser()
	
	in_group = parser.add_mutually_exclusive_group(required=True)
	in_group.add_argument("-i", "--input-bin", type=str, help="input file containing the binary bitstream")
	in_group.add_argument("--hdf5", type=str, help="HDF5 containing one or multiple input bitstreams")
	
	parser.add_argument("--input-path", type=str, help="path inside the HDF5 file for the input bitstream; if group, all datasets inside will be processed")
	parser.add_argument("--result-path", type=str, help="path inside the HDF5 file for the result")
	
	parser.add_argument("-c", "--chip-type", default="8k", type=str, choices=["8k"], help="type of FPGA")
	parser.add_argument("-b", "--bram-banks", nargs="*", type=int, help="BRAM banks to be included in the bitstream")
	parser.add_argument("-l", "--level", default=0, type=int, help="optimization level")
	parser.add_argument("-s", "--skip-comment", action="store_true", help="skip the comment at the beginning of the output")
	parser.add_argument("--bram-chunk-size", default=128, type=int, help="maximum size of BRAM that will be written in one go")
	parser.add_argument("-d", "--description", type=str, help="description of the output to be added to HDF5 attributes")
	
	sub_parsers = parser.add_subparsers()
	rewrite_parser = sub_parsers.add_parser("rewrite", help="rewrite the bitstream according to the other arguments")
	rewrite_parser.add_argument("-o", "--output-bin", type=str, help="output file for the binary bitstream")
	rewrite_parser.set_defaults(function=bitstream_from_args)
	
	benchmark_parser = sub_parsers.add_parser("benchmark", help="measure the configuration time of the bitstream according to the arguments")
	benchmark_parser.add_argument("-n", "--number", default=1, type=int, help="number of times the configuration is done during one measurement")
	benchmark_parser.add_argument("-r", "--repeat", default=10, type=int, help="number of measurements taken")
	benchmark_parser.set_defaults(function=benchmark)
	
	return parser

if __name__ == "__main__":
	arg_parser = create_arg_parser()
	in_args = arg_parser.parse_args()
	run(in_args)

