#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <cstdarg>
#include "log.h"

class Setting
{
public:
	std::string key;
	std::string value;

	inline bool isTrue() const { return value == "yes" || value == "true"; }
	inline bool isFalse() const { return !isTrue(); }
	inline uint8_t u8() const { return std::stoi(value) & 0xFF; }
	inline uint16_t u16() const { return std::stoi(value) & 0xFFFF; }
	inline uint32_t u32() const { return std::stoi(value); }
	inline float f32() const { return std::stof(value); }
	inline const std::string& str() { return value; }
};

class Settings : public std::map<std::string, Setting>
{
public:
	void open(const std::string& path);
	void save(const std::string& path);

	bool isTrue(const std::string& fmt, ...);
	bool isFalse(const std::string& fmt, ...);
	uint8_t u8(const std::string& fmt, ...);
	uint16_t u16(const std::string& fmt, ...);
	uint32_t u32(const std::string& fmt, ...);
	float f32(const std::string& fmt, ...);
	const std::string& str(const std::string& fmt, ...);
};
