#include <cufft.h>
#include <ncurses.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "conv.h"

#ifndef NUM_CONV_INSTANCES
#define NUM_CONV_INSTANCES 1 
#endif

#ifndef NUM_SKIP_INPUTS
#define NUM_SKIP_INPUTS 0
#endif

const wchar_t* logolines[] = {
	L"██     ██ ██ ██████  ██   ██  █████  ████████      ██████ ██    ██ ██████   █████   ██████  ██████  ███    ██ ██    ██ ",
	L"██     ██ ██ ██   ██ ██  ██  ██   ██    ██        ██      ██    ██ ██   ██ ██   ██ ██      ██    ██ ████   ██ ██    ██ ",
	L"██  █  ██ ██ ██████  █████   ███████    ██        ██      ██    ██ ██   ██ ███████ ██      ██    ██ ██ ██  ██ ██    ██ ",
	L"██ ███ ██ ██ ██      ██  ██  ██   ██    ██        ██      ██    ██ ██   ██ ██   ██ ██      ██    ██ ██  ██ ██  ██  ██  ",
	L" ███ ███  ██ ██      ██   ██ ██   ██    ██         ██████  ██████  ██████  ██   ██  ██████  ██████  ██   ████   ████   ",
};

int display(JackClient* c)
{
	setlocale(LC_ALL, "");
	initscr();
	if (!has_colors())
	{
		endwin();
		Log::error(__func__, "Terminal does not have colors");
		return 1;
	}
	if (!can_change_color())
	{
		endwin();
		Log::error(__func__, "Unable to change colors, probably need to `export TERM=xterm-256color`");
		return 2;
	}
	start_color();

	init_color(1, 800, 100, 400);
	init_color(2, 800, 100, 500);
	init_color(3, 800, 100, 600);
	init_color(4, 800, 100, 700);
	init_color(5, 800, 100, 800);
	init_color(6, 600, 300, 100);
	init_color(7, 700, 200, 100);
	init_color(8, 800, 100, 100);
	init_color(9, 500, 500, 500);
	init_color(10, 100, 100, 100);
	init_color(11, 150, 150, 150);

	init_pair(1, 1, 0);
	init_pair(2, 2, 0);
	init_pair(3, 3, 0);
	init_pair(4, 4, 0);
	init_pair(5, 5, 0);
	init_pair(6, 6, 0);
	init_pair(7, 7, 0);
	init_pair(8, 8, 0);
	init_pair(9, 9, 8);
	init_pair(10, 9, 10);
	init_pair(11, 9, 11);

	cbreak();
	keypad(stdscr, TRUE);
	noecho();

	size_t rows, cols;
	getmaxyx(stdscr, rows, cols);

	int i = 0;
	for (int l = 0; l < 5; l++)
	{
		attron(COLOR_PAIR(1 + l));
		mvaddwstr(1+l, 1, logolines[l]);
	}

	auto midiports = jack_get_ports(c->handle, NULL, JACK_DEFAULT_AUDIO_TYPE, 0);
	for (auto midiport = midiports; *midiport; midiport++)
	{
		char* client = (char*)alloca(strlen(*midiport) + 1);
		strcpy(client, *midiport);
		char* port = strchr(client, ':');
		*port = 0;
		port++;
		
		attron(COLOR_PAIR(10 + (i & 1)));
		mvprintw(10+ ++i, 10, "%-12s | %-24s", client, port);
	}
	refresh();

	for (int ch, running = 1; running && (ch = getch()); )
	{
		switch (ch)
		{
		case 'q': 
		case KEY_F(10):
			running = 0;
			break;
		};
	}
	endwin();
	return 0;
}

int main()
{
	selectGpu();


	// Top row of my novation launchcontrol starts at 0x15
	uint8_t ccMessage = 0xB0;
	uint8_t ccStart = 0x15;

	Convolution* instances[NUM_CONV_INSTANCES] = { nullptr };
	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		char* name = (char*)alloca(256);
		sprintf(name, "cudaconv_%lu",i+1);

		// There are 4 controls, let's assume simply that cc is contiguous
		// Other mappings would require changing Convolution::cc member
		auto c = instances[i] = new Convolution(name, ccMessage + i, ccStart);
	
		std::ifstream is("index.txt");
		std::string path;
		for (size_t idx = 0; std::getline(is, path); idx++)
		{
			WavFile w(path);
			c->prepare(idx, w);
		}
		c->start();

		// Auto connect to capture_<i+1>, there are nicer ways to do this but connecting ports
		// is going to be dealt with later in different code.
		#if 1
		sprintf(name, "system:capture_%lu", i+1+NUM_SKIP_INPUTS);
		jack_connect(c->handle, name, jack_port_name(c->input));
		#else
		jack_connect(c->handle, "system:capture_1", jack_port_name(c->input));
		#endif
		
		// Connect to stereo output, assumed to be available
		jack_connect(c->handle, jack_port_name(c->left),  "system:playback_1");
		jack_connect(c->handle, jack_port_name(c->right), "system:playback_2");

		// Auto connect all MIDI ports
		#if 1
		auto midiports = jack_get_ports(c->handle, NULL, JACK_DEFAULT_MIDI_TYPE, JackPortIsOutput);
		for (auto midiport = midiports; *midiport; midiport++)
		{
			Log::info(__func__, "Found MIDI port: %s", *midiport);
			jack_connect(c->handle, *midiport, jack_port_name(c->midiIn));
		}
		jack_free(midiports);
		#endif

	}

	int rc = display(instances[0]);

	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		auto c = instances[i];
		if (c->isRunning()) c->stop();
		Log::info(c->name, "Average convolution runtime: %f", c->avgRuntime());
		delete c;
	}

	return rc;
}
