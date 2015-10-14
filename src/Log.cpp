#include "Log.h"

LogLevel Logger::level_ = LogLevel::Info;
std::FILE* Logger::file_ = nullptr;

Logger::Logger()
{
	level_ = LogLevel::Info;
	file_ = nullptr;
}

Logger::~Logger()
{
	CloseLogFile();
}

void Logger::Reset(std::string filename, LogLevel level)
{
	level_ = level;
	file_ = nullptr;
	ResetLogFile(filename);
}

int Logger::ResetLogFile(std::string filename)
{
	// close the current log file
	CloseLogFile();
	// If the filename is specified, try to open it, or just write the
	// messages to standard output if filename is empty or openning fail.
	if (filename.size() > 0)
	{
		file_ = fopen(filename.c_str(), "w");
		if (file_ == nullptr) // fail on openning file
		{
			Printf(LogLevel::Error, "Cannot create log file %s\n",
				filename.c_str());
			return -1;
		}
	}
	return 0;
}

void Logger::ResetLogLevel(LogLevel level)
{
	level_ = level;
}

int Logger::Printf(LogLevel level, const char *format, ...)
{
	// omit the message with low level
	if (level < level_)
	{
		return 0;
	}

	std::string level_str = GetLevelStr(level);
	std::string time_str = GetSystemTime();
	va_list val;
	va_start(val, format);
	// write the message to standard output
	printf("[%s] [%s] ", level_str.c_str(), time_str.c_str());
	int ret = vprintf(format, val);
	fflush(stdout);
	// write the message to log file
	if (file_ != nullptr)
	{
		fprintf(file_, "[%s] [%s] ", level_str.c_str(), time_str.c_str());
		vfprintf(file_, format, val);
		fflush(file_);
	}
	va_end(val);

	// If it is a FATAL error, kill the process
	if (LogLevel::Fatal == level)
	{
		CloseLogFile();
		exit(1);
	}

	return ret;
}

void Logger::CloseLogFile()
{
	if (file_ != nullptr)
	{
		fclose(file_);
		file_ = nullptr;
	}
}

std::string Logger::GetSystemTime()
{
	time_t t = time(0);
	char str[64];
	strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", localtime(&t));
	return str;
}

std::string Logger::GetLevelStr(LogLevel level)
{
	switch (level)
	{
	case LogLevel::Debug: return "DEBUG";
	case LogLevel::Info: return "INFO";
	case LogLevel::Error: return "ERROR";
	case LogLevel::Fatal: return "FATAL";
	default: return "UNKNOW";
	}
}