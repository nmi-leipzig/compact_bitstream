from dataclasses import dataclass
from typing import NamedTuple, Tuple, Dict
import enum

TileType = enum.Enum("TileType", ["LOGIC", "IO", "RAM_T", "RAM_B"])

class TilePosition(NamedTuple):
	x: int
	y: int

class Bit(NamedTuple):
	group: int
	index: int

class ExtraBit(NamedTuple):
	"""Bit in the CRAM that is not associated with a tile"""
	bank: int
	x: int
	y: int

class BRAMIndicator(NamedTuple):
	"""Indicator for usage of certain BRAM block"""
	offset: TilePosition
	bit: Bit
	value: bool

@dataclass(frozen=True)
class DeviceSpec:
	asc_name: str
	
	max_x: int
	max_y: int
	cram_width: int
	cram_height: int
	bram_width: int
	bram_height: int
	
	bram_cols: Tuple[int, ...]
	bram_indicator: BRAMIndicator
	tile_type_width: Dict[TileType, int]
	extra_bits: Tuple[ExtraBit, ...]
	
	tile_height: int = 16
	
	def get_tile_types(self):
		# IO
		for y in range(1, self.max_y):
			yield TilePosition(0, y), TileType.IO
			yield TilePosition(self.max_x, y), TileType.IO
		for x in range(1, self.max_x):
			yield TilePosition(x, 0), TileType.IO
			yield TilePosition(x, self.max_y), TileType.IO
		
		# RAM & LOGIC
		for x in range(1, self.max_x):
			for y in range(1, self.max_y):
				if x in self.bram_cols:
					if y%2 == 0:
						ttype = TileType.RAM_T
					else:
						ttype = TileType.RAM_B
				else:
					ttype = TileType.LOGIC
				yield TilePosition(x, y), ttype
	

SPECS = (
	DeviceSpec(
		"8k",
		33,
		33,
		872,
		272,
		128,
		256,
		(8, 25),
		BRAMIndicator(TilePosition(0, 0), Bit(1, 7), True),
		{TileType.IO: 18, TileType.RAM_T: 42, TileType.RAM_B: 42, TileType.LOGIC: 54},
		(
			ExtraBit(0, 870, 270), ExtraBit(0, 871, 270), ExtraBit(1, 870, 271), ExtraBit(1, 871, 271),
			ExtraBit(1, 870, 270), ExtraBit(1, 871, 270), ExtraBit(0, 870, 271), ExtraBit(0, 871, 271),
		),
	),
	DeviceSpec(
		"1k",
		13,
		17,
		332,
		144,
		64,
		256,
		(3, 10),
		BRAMIndicator(TilePosition(0, 0), Bit(1, 7), False),
		{TileType.IO: 18, TileType.RAM_T: 42, TileType.RAM_B: 42, TileType.LOGIC: 54},
		(
			ExtraBit(0, 330, 142), ExtraBit(0, 331, 142), ExtraBit(1, 330, 143), ExtraBit(1, 331, 143),
			ExtraBit(1, 330, 142), ExtraBit(1, 331, 142), ExtraBit(0, 330, 143), ExtraBit(0, 331, 143),
		),
	),
)

SPECS_BY_ASC = {d.asc_name: d for d in SPECS}
