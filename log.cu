#include "log.h"

#include <cstdarg>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

void Log::message(std::ostream& os, const char* style, const char* type, const std::string& id, const char* msg) noexcept
{
	auto now = std::chrono::system_clock::now();
	auto dt = std::chrono::system_clock::to_time_t(now);
	os << ESC(37;2) << type << " "
	   << std::put_time(std::localtime(&dt), "%F %T") << ESC(0) << ESC(37)
	   << style << ESC(1;2) " [" << id << ESC(0) << style << ESC(1;2) "] " ESC(0) << ESC(37);

	os << style << msg << EOL << std::flush;
}

void Log::info(const std::string& id, const char* fmt, ... ) noexcept
{
	int rc;
	char buffer[256];
	
	assert(fmt);	
	va_list args;
	va_start(args, fmt);
	rc = vsnprintf(buffer, 255, fmt, args);
	assert(0 <= rc);
	va_end(args);
	assert(rc > 0);
		
	message(std::cout, ESC(37), "I", id, buffer); 
}

void Log::warn(const std::string& id, const char* fmt, ... ) noexcept
{
	int rc;
	char buffer[256];

	assert(fmt);	
	va_list args;
	va_start(args, fmt);
	rc = vsnprintf(buffer, 255, fmt, args);
	assert(0 <= rc);
	va_end(args);
	assert(rc > 0);

	message(std::cerr, ESC(33;1), ESC(33) "W", id, buffer); 
}

void Log::error(const std::string& id, const char* fmt, ... ) noexcept
{
	int rc;
	char buffer[256];

	assert(fmt);	
	va_list args;
	va_start(args, fmt);
	rc = vsnprintf(buffer, 255, fmt, args);
	assert(0 <= rc);
	va_end(args);
	assert(rc > 0);
		
	message(std::cerr, ESC(31;1;2), ESC(31;1) "E", id, buffer); 
}

void Log::newline() noexcept
{
	std::cout << std::string(22, ' ');
	std::cout << std::endl;
}

void Log::newline(const char* fmt, ...) noexcept
{
	int rc;
	char buffer[256];

	assert(fmt);	
	va_list args;
	va_start(args, fmt);
	rc = vsnprintf(buffer, 255, fmt, args);
	assert(0 <= rc);
	va_end(args);
	assert(rc > 0);
		
	std::cout << std::string(22, ' ');
	std::cout << buffer;
	std::cout << std::endl;
}

