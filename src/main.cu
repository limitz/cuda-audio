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
//123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 |
L"       ██████ ██    ██ ██████   █████    ██████  ██████  ███    ██ ██    ██     ",
L"      ██      ██    ██ ██   ██ ██   ██  ██      ██    ██ ████   ██ ██    ██     ",
L"      ██      ██    ██ ██   ██ ███████  ██      ██    ██ ██ ██  ██ ██    ██     ",
L"      ██      ██    ██ ██   ██ ██   ██  ██      ██    ██ ██  ██ ██  ██  ██      ",
L"       ██████  ██████  ██████  ██   ██   ██████  ██████  ██   ████   ████       ",
L"       W  I  P  K  A  T                https://github.com/limitz/cudaconv       ",
L"                                                                                ",
L"  1   2   3   4   5   6   7   8   < IMPULSE RESPONSE >                    OUT   ",
L"+---+---+---+---+---+---+---+---+--------------------------------------+-------+",
L"|   |   |   |   |   |   |   |   |                                    |         |",
L"|   |   |   |   |   |   |   |   |   B015 > 1.Glass Bottle Hall       |         |",
L"|   |   |   |   |   |   |   |   |                                    |         |",
L"|   |   |   |   |   |   |   |   |   B017 > DRY: [##########] 100     |         |",
L"|   |   |   |   |   |   |   |   |   B018 > WET: [###       ]  30     |         |",
L"|   |   |   |   |   |   |   |   |   B016 > PRE: [#######   ]  70     |         |",
L"|   |   |   |   |   |   |   |   |                                    |         |",
L"+----+----+----+----+----+----+----+----+------------------------------+-------+",
L"|  06:41  | CUDACONV started...                                                |",
L"|  06:42  |                                                                    |",
L"|         |                                                                    |",
L"|         |                                                                    |",
L"|         |                                                                    |",
L"|         |                                                                    |",
L"+---------+--------------------------------------------------------------------+",
L"  Q: Quit   F<n>: Select C<n>", 
};

void drawSingleMeter(int n, float v)
{
	for (int i=0; i<7; i++)
	{
		attron(COLOR_PAIR(30 + i));
		mvaddwstr(16-i, n * 4 + 1, v >= i / 6.0f ? L"=== " : L"    ");
	}
}

int display(JackClient* c)
{
	setlocale(LC_ALL, "");
	initscr();
	if (!has_colors())
	{
		endwin();
		Log::error(__func__, "Terminal does not have colors");
		std::cin.get();
		return 0;
	}
	if (!can_change_color())
	{
		endwin();
		Log::error(__func__, "Unable to change colors, probably need to `export TERM=xterm-256color`");
		std::cin.get();
		return 0;
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
	init_color(9, 900, 900, 200);
	init_color(10, 100, 100, 100);
	init_color(11, 500, 500, 500);
	init_color(12, 300, 1000,   0);
	init_color(13, 600,  900,   0);
	init_color(14, 800,  800,   0);
	init_color(15,1000,  600,   0);
	init_color(16,1000,    0, 200);
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
	init_pair(20, 9, 0);
	init_pair(21, 11, 0);
	init_pair(30, 12, 0);
	init_pair(31, 12, 0);
	init_pair(32, 12, 0);
	init_pair(33, 13, 0);
	init_pair(34, 14, 0);
	init_pair(35, 15, 0);
	init_pair(36, 16, 0);

	cbreak();
	keypad(stdscr, TRUE);
	noecho();

	size_t rows, cols;
	getmaxyx(stdscr, rows, cols);

	int i = 0;
	for (int l = 0; l < 5; l++)
	{
		attron(COLOR_PAIR(1 + l));
		mvaddwstr(1+l, 0, logolines[l]);
	}
	for (int l = 5; l < 7; l++)
	{
		attron(COLOR_PAIR(20));
		mvaddwstr(1+l, 0, logolines[l]); 
	}

	for (int l = 7; l < sizeof(logolines) / sizeof(*logolines); l++)
	{
		attron(COLOR_PAIR(21));
		mvaddwstr(1+l, 0, logolines[l]);
	}
	
	for (int i=0; i<8; i++)
	{
		drawSingleMeter(i, 1.0f);
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

		// There are 5 controls, let's assume simply that cc is contiguous
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
