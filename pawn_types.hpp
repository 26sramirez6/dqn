#ifndef PAWN_TYPES_HPP
#define PAWN_TYPES_HPP

#include <unordered_map>
#include <string>
#include "lux/kit.hpp"

static inline const lux::Cell* get_destination_cell(
	const lux::GameMap& _map, const lux::Unit& _unit, const int _action) {
	int newx = _unit.pos.x;
	int newy = _unit.pos.y;
	switch (_action){
	case (WorkerActionInt::north):
		newy -= 1;
		break;
	case (WorkerActionInt::east):
		newx += 1;
		break;
	case (WorkerActionInt::south):
		newy += 1;
		break;
	case (WorkerActionInt::west):
		newx -= 1;
		break;
	default:
		break;
	}

	const lux::Cell * destination_cell = _map.getCell(newx, newy);
	return destination_cell;
}

struct Worker{
	using type = lux::Unit;
	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		return std::stoi(++++cstr);
	}
		
	static inline std::vector<int> get_pawn_ids(const kit::Agent& _agent) {
		const auto& player = _agent.players[_agent.id];
		const auto& units = player.units;
		std::vector<int> ids;
		ids.reserve(units.size());
		for (int i = 0; i < units.size(); i++) {
			if (units[i].isWorker()) ids.push_back(string_to_id(units[i].id));
		}
		return ids;
	}

	static inline std::unordered_map<int, lux::Unit> 
	get_pawns(const kit::Agent& _agent) {
		std::unordered_map<int, lux::Unit> pawn_map;
		const auto& player = _agent.players[_agent.id];
		const auto& units = player.units;
		for (int i = 0; i < units.size(); i++) {
			if (units[i].isWorker()) {
				pawn_map.insert({string_to_id(units[i].id), units[i]});
			}
		}
		return pawn_map;
	}

	static inline void
	clean_actions(const kit::Agent &_agent, Eigen::Ref<Eigen::ArrayXi> actions_) {
		const auto& player = _agent.players[_agent.id];
		const auto& map = _agent.map;
		const auto& units = player.units;
		const auto& opponent = _agent.players[_agent.id + 1 %2];
		const bool opponent_has_cities = opponent.cities.size() > 0;
		// assumes units are ordered same as actions
		for (int i = 0; i < units.size(); ++i) {
			const auto& unit = units[i];
			int& action = actions_(i);
						
			if (!unit.canAct() ||
					(action==WorkerActionInt::build) || //REMOVE
					(action==WorkerActionInt::build && !unit.canBuild(_agent.map)) ||
					(unit.pos.x==0 && action==WorkerActionInt::west) ||
					(unit.pos.x==BoardConfig::size-1 && action==WorkerActionInt::east) ||
					(unit.pos.y==0 && action==WorkerActionInt::north) ||
					(unit.pos.y==BoardConfig::size-1 && action==WorkerActionInt::south) 
				) {
				action = WorkerActionInt::center;
				continue;
			} 
			
			if (opponent_has_cities) {
				const lux::Cell * destination_cell = get_destination_cell(map, unit, action);

				if (destination_cell->citytile!=nullptr && 
						destination_cell->citytile->team!=_agent.id) {
					action = WorkerActionInt::center;
					continue;
				}

			}

		}
	}
};

struct CityTile {
	using type = lux::CityTile;
	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		std::stoi(++++cstr);
	}

//	static inline std::vector<int> getPawnIds(const kit::Agent& _agent) {
//		const auto& player = _agent.players[_agent.id];
//		const auto& cities = player.cities;
//		std::vector<int> ids;
//		for (auto& kv : cities) {
//			const auto& ctiles = kv.second.citytiles;
//			for (int i = 0; i < ctiles.size(); i++) {
//				ids.push_back(ctiles[i]);
//			}
//		} 
//	}

};

struct Cart {
	using type = lux::Unit;
	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		return std::stoi(++++cstr);
	}

	static inline std::vector<int> getPawnIds(const kit::Agent& _agent) {
		const auto& player = _agent.players[_agent.id];
		const auto& units = player.units;
		std::vector<int> ids;
		ids.reserve(units.size());
		for (int i = 0; i < units.size(); i++) {
			if (!units[i].isWorker()) ids.push_back(string_to_id(units[i].id));
		}
		return ids;
	}
};
#endif
