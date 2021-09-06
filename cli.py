#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace

from src.configuration import BinOpt, Configuration

def compact(args: Namespace) -> None:
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
	
	with open(args.output_bin, "wb") as bin_file:
		config.write_bin(bin_file, opt=opt)

def create_arg_parser() -> ArgumentParser:
	parser = ArgumentParser()
	
	parser.add_argument("-c", "--chip-type", default="8k", type=str, choices=["8k"], help="type of FPGA")
	parser.add_argument("-i", "--input-bin", required=True, type=str, help="input file containing the binary bitstream")
	parser.add_argument("-o", "--output-bin", required=True, type=str, help="output file for the binary bitstream")
	parser.add_argument("-b", "--bram-banks", nargs="*", type=int, help="BRAM banks to be included in the bitstream")
	parser.add_argument("-l", "--level", default=0, type=int, help="optimization level")
	parser.add_argument("-s", "--skip-comment", action="store_true", help="skip the comment at the beginning of the output")
	parser.add_argument("--bram-chunk-size", default=128, type=int, help="maximum size of BRAM that will be written in one go")
	
	return parser

if __name__ == "__main__":
	arg_parser = create_arg_parser()
	in_args = arg_parser.parse_args()
	compact(in_args)
