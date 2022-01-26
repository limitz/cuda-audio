#include "settings.h"
#include <cassert>

void Settings::open(const std::string& path)
{
	std::ifstream is(path, std::ifstream::binary);
	while (!is.eof())
	{
		std::string key, value;
		is >> key;
		if (!key.size()) break;
		
		if ('#' == key[0]) 
		{
			std::string comment;
			std::getline(is, comment);
			continue;
		}
		is >> value;
		(*this)[key] = Setting { key, value };
		Log::info("Settings", "%-24s " ESC(36;1) "%s", key.c_str(), value.c_str());
	}
	is.close();
}

void Settings::save(const std::string& path)
{
	assert(false && "Not implemented");
}

#define SETTINGS_FORMAT_KEY(k, fmt) \
	va_list k ## _va_list; \
	va_start(k ## _va_list, fmt); \
	auto k ## _n = vsnprintf(nullptr, 0, fmt.c_str(), k ## _va_list); \
	va_end(k ## _va_list);\
	assert(0 < k ## _n); \
	auto k ## _buf = (char*) alloca(k ## _n + 1); \
	va_start(k ## _va_list, fmt); \
	vsnprintf(k ## _buf, k ## _n + 1, fmt.c_str(), k ## _va_list);\
	auto k = std::string(k ## _buf); \
	va_end(k ## _va_list);

	
bool Settings::isTrue(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].isTrue();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

bool Settings::isFalse(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].isFalse();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

uint8_t Settings::u8(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].u8();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

uint16_t Settings::u16(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].u16();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

uint32_t Settings::u32(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].u32();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

float Settings::f32(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].f32();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}

const std::string& Settings::str(const std::string& fmt, ...)
{
	SETTINGS_FORMAT_KEY(key, fmt);
	try
	{
		return (*this)[key].str();
	}
	catch (std::exception& e)
	{
		Log::error("Settings", "Error for key %s", key.c_str());
		throw e;
	}
}
