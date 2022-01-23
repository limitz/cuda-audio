#include <cufft.h>
#include <ncurses.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "gpu.h"
#include "wav.h"
#include "jackclient.h"
#include "conv.h"

#ifndef NUM_CONV_INSTANCES
#define NUM_CONV_INSTANCES 1 
#endif

// TODO: remove after refactoring, extern decl in conv.cu
WavFile* wav[38];

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

	wav[0] = new WavFile("ir/1/Block Inside.wav");
	wav[1] = new WavFile("ir/1/Bottle Hall.wav");
	wav[2] = new WavFile("ir/1/Cement Blocks 1.wav");
	wav[3] = new WavFile("ir/1/Cement Blocks 2.wav");
	wav[4] = new WavFile("ir/1/Chateau de Logne, Outside.wav");
	wav[5] = new WavFile("ir/1/Conic Long Echo Hall.wav");
	wav[6] = new WavFile("ir/1/Deep Space.wav");
	wav[7] = new WavFile("ir/1/Derlon Sanctuary.wav");
	wav[8] = new WavFile("ir/1/Direct Cabinet N1.wav");
	wav[9] = new WavFile("ir/1/Direct Cabinet N2.wav");
	wav[10] = new WavFile("ir/1/Direct Cabinet N3.wav");
	wav[11] = new WavFile("ir/1/Direct Cabinet N4.wav");
	wav[12] = new WavFile("ir/1/Five Columns.wav");
	wav[13] = new WavFile("ir/1/Five Columns Long.wav");
	wav[14] = new WavFile("ir/1/French 18th Century Salon.wav");
	wav[15] = new WavFile("ir/1/Going Home.wav");
	wav[16] = new WavFile("ir/1/Greek 7 Echo Hall.wav");
	wav[17] = new WavFile("ir/1/Highly Damped Large Room.wav");
	wav[18] = new WavFile("ir/1/In The Silo.wav");
	wav[19] = new WavFile("ir/1/In The Silo Revised.wav");
	wav[20] = new WavFile("ir/1/Large Bottle Hall.wav");
	wav[21] = new WavFile("ir/1/Large Long Echo Hall.wav");
	wav[22] = new WavFile("ir/1/Large Wide Echo Hall.wav");
	wav[23] = new WavFile("ir/1/Masonic Lodge.wav");
	wav[24] = new WavFile("ir/1/Musikvereinsaal.wav");
	wav[25] = new WavFile("ir/1/Narrow Bumpy Space.wav");
	wav[26] = new WavFile("ir/1/Nice Drum Room.wav");
	wav[27] = new WavFile("ir/1/On a Star.wav");
	wav[28] = new WavFile("ir/1/Parking Garage.wav");
	wav[29] = new WavFile("ir/1/Rays.wav");
	wav[30] = new WavFile("ir/1/Right Glass Triangle.wav");
	wav[31] = new WavFile("ir/1/Ruby Room.wav");
	wav[32] = new WavFile("ir/1/Scala Milan Opera Hall.wav");
	wav[33] = new WavFile("ir/1/Small Drum Room.wav");
	wav[34] = new WavFile("ir/1/Small Prehistoric Cave.wav");
	wav[35] = new WavFile("ir/1/St Nicolaes Church.wav");
	wav[36] = new WavFile("ir/1/Trig Room.wav");
	wav[37] = new WavFile("ir/1/Vocal Duo.wav");

	// Top row of my novation launchcontrol starts at 0x15
	uint8_t ccMessage = 0xB0;
	uint8_t ccStart = 0x15;

	Convolution* instances[NUM_CONV_INSTANCES];
	for (auto i=0UL; i < NUM_CONV_INSTANCES; i++)
	{
		char* name = (char*)alloca(256);
		sprintf(name, "cudaconv_%lu",i+1);

		// There are 4 controls, let's assume simply that cc is contiguous
		// Other mappings would require changing Convolution::cc member
		auto c = instances[i] = new Convolution(name, ccMessage + i, ccStart);
		c->start();

		// Auto connect to capture_<i+1>, there are nicer ways to do this but connecting ports
		// is going to be dealt with later in different code.
		#if 1
		sprintf(name, "system:capture_%lu", i+1);
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


	for (auto i=0UL; i<sizeof(wav)/sizeof(*wav); i++) delete wav[i];
	return rc;
}
