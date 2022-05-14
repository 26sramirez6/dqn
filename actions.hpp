#ifndef ACTIONS_HPP_
#define ACTIONS_HPP_

enum class WorkerActions {
	CENTER,
	NORTH,
	EAST,
	SOUTH,
	WEST,
	//PILLAGE,
	//TRANSFER,
	BUILD,
	Count
};

struct WorkerActionInt {
	static constexpr int center = 0;
	static constexpr int north = 1;
	static constexpr int east = 2;
	static constexpr int south = 3;
	static constexpr int west = 4;
	static constexpr int build = 5;
};

enum class CartActions {
	CENTER,
	NORTH,
	EAST,
	SOUTH,
	WEST,
	TRANSFER,
	Count
};

enum class CityTileActions {
	NONE,
	BUILD_WORKER,
	//BUILD_CART,
	RESEARCH,
	Count
};

#endif
