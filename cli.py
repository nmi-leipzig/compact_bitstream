#!/usr/bin/env python3

import statistics
import timeit

from argparse import ArgumentParser, Namespace

from src.configuration import BinOpt, Configuration
from src.fpga_board import FPGABoard

def bitstream_from_args(args: Namespace) -> bytes:
	config = Configuration.create_blank(args.chip_type)
	
	with open(args.input_bin, "rb") as bin_file:
		config.read_bin(bin_file)
	
	opt = BinOpt(
		bram_chunk_size = args.bram_chunk_size,
		detect_used_bram = False,
		bram_banks = args.bram_banks,
		optimize = args.level,
		skip_comment = args.skip_comment,
	)
	
	return config.get_bitstream(opt)

def rewrite(args: Namespace) -> None:
	bitstream = bitstream_from_args(args)
	
	with open(args.output_bin, "wb") as bin_file:
		bin_file.write(bitstream)

def benchmark(args: Namespace) -> None:
	bitstream = bitstream_from_args(args)
	
	with FPGABoard.get_suitable_board() as fpga:
		res = timeit.repeat("fpga.flash_bitstream(bitstream)", repeat=args.repeat, number=args.number, globals={"fpga": fpga, "bitstream": bitstream})
	
	print(res)
	print(f"stdev: {statistics.stdev(res)}, mean: {statistics.mean(res)}")

def create_arg_parser() -> ArgumentParser:
	parser = ArgumentParser()
	
	parser.add_argument("-c", "--chip-type", default="8k", type=str, choices=["8k"], help="type of FPGA")
	parser.add_argument("-i", "--input-bin", required=True, type=str, help="input file containing the binary bitstream")
	parser.add_argument("-b", "--bram-banks", nargs="*", type=int, help="BRAM banks to be included in the bitstream")
	parser.add_argument("-l", "--level", default=0, type=int, help="optimization level")
	parser.add_argument("-s", "--skip-comment", action="store_true", help="skip the comment at the beginning of the output")
	parser.add_argument("--bram-chunk-size", default=128, type=int, help="maximum size of BRAM that will be written in one go")
	
	sub_parsers = parser.add_subparsers()
	rewrite_parser = sub_parsers.add_parser("rewrite", help="rewrite the bitstream according to the other arguments")
	rewrite_parser.add_argument("-o", "--output-bin", required=True, type=str, help="output file for the binary bitstream")
	rewrite_parser.set_defaults(function=rewrite)
	
	benchmark_parser = sub_parsers.add_parser("benchmark", help="measure the configuration time of the bitstream according  to the arguments")
	benchmark_parser.add_argument("-n", "--number", default=1, type=int, help="number of times the configuration is done during one measurement")
	benchmark_parser.add_argument("-r", "--repeat", default=10, type=int, help="number of measurements taken")
	benchmark_parser.set_defaults(function=benchmark)
	
	return parser

if __name__ == "__main__":
	arg_parser = create_arg_parser()
	in_args = arg_parser.parse_args()
	in_args.function(in_args)
