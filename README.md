# compact_bitstream

Reduces the size of bitstreams for iCE40 HX8K FPGAs.

There are two subcommands:
- `rewrite`: compact bitstream
- `benchmark`: measure reconfiguration time of compacted bitstream

## Compacting a Bitstream
`./comp_cli.py -i uncompressed.bin -b 0 1 -s -l 4 rewrite -o compressed.bin`\
`-i`: input bitstream\
`-b`: BRAM banks to include in the compacted bitstream, banks 0 and 1 in the example (BRAM block skipping)\
`-s`: remove comment (comment skipping)\
`-l`: compaction level, 5 is highest\
`rewrite`: subcommand to rewrite a bitstream\
`-o`: output bitstream

## Benchmarking a Bitstream
`./comp_cli.py --hdf5 results.h5 --input-path "/bitstream/in/hdf5" --result-path "/result/in/hdf5" -b -s -l 3 -d "comment added to HDF5" benchmark -r 10000`
`--hdf5`: HDF5 file\
`--input-path`: path in the HDF5 file to the original bitstream; it will be compacted before benchmarking\
`--result-path`: path in the HDF5 file where the time measurements will be stored\
`-b`: BRAM banks to include in the compacted bitstream, no bank in the example (BRAM block skipping)\
`-s`: remove comment (comment skipping)\
`-l`: compaction level, 4 means no chunk sorting\
`-d`: descriptive comment that will be added to the time measurements in the HDF5 file\
`benchmark`: subcommand to benchmark a bitstream\
`-r`: number of measurements taken

## Compaction levels
- 0: no optimization, write whole CRAM, one chunk per bank
- 1: set bank width, height, number and offset only if differs from currently set value (value persistence)
- 2: in addition to 1, skip CRAM banks that only contain zeros
- 3: in addition to 2, skip CRAM lines that only contain zeros (zero row skipping)
- 4: in addition to 3, order writes by chunk size to reduce number of set height commands (chunk sorting)
