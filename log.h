#pragma once
#include <iostream>
#include <string>

#ifdef ESC
#error ESC already defined
#else
#define ESC(n) "\x1b[" # n "m"
#endif

#ifdef EOL
#error EOL already defined
#else
#define EOL ESC(0) "\n"
#endif

inline std::string escapeRgb(uint8_t r, uint8_t g, uint8_t b)
{
	char* buffer = (char*)alloca(32);
	b = pow(b / 256.,0.5) * 0xff;
	g = pow(g / 256.,0.5) * 0xff ;
	r = pow(r / 256.,0.5) * 0xff ;

	sprintf(buffer, "\x1b[38;2;%d;%d;%dm", r, g, b);
	return std::string(buffer);
}

class Log
{
protected:
	static void message(
			std::ostream& os, 
			const char* style, 
			const char* type, 
			const std::string& id, 
			const char* msg) noexcept;

public:
	static void info  (const std::string& id, const char* fmt, ... ) noexcept;
	static void warn  (const std::string& id, const char* fmt, ... ) noexcept;
	static void error (const std::string& id, const char* fmt, ... ) noexcept;
	static void newline() noexcept;
	static void newline(const char* fmt, ...) noexcept;

	static void lock() noexcept;
	static void unlock() noexcept;
};

