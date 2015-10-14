#pragma once

#include <fstream>
#include <string>
#include <cstdarg>
#include <ctime>

/*!
* \brief A enumeration type of log message levels.
* \note The values are ordered: DEBUG < INFO < ERROR < FATAL.
*/
enum class LogLevel : int
{
	Debug = 0,
	Info = 1,
	Error = 2,
	Fatal = 3
};


/*!
* \brief The class Logger is responsible for writing log messages into
* standard output or log file.
*/
class Logger
{
public:
	/*!
	* \brief Creates an instance of class Logger.
	*
	* By default, the log messages will be written to standard output with
	* minimal level of INFO. Users are able to further set the log file or
	* log level with corresponding methods.
	*/
	Logger();
	~Logger();

	/*!
	* \brief Reset the setting of the Logger by specifying log file
	*        and log level.
	*
	* The log message will be written to both standard output and file (if
	* created successfully).
	* \param filename Log file name
	* \param level Log level
	*/
	static void Reset(std::string filename, LogLevel level = LogLevel::Info);

	/*!
	* \brief Resets the log file.
	* \param filename The new log filename. If it is empty, the Logger
	*        will close current log file (if it exists).
	*/
	static int ResetLogFile(std::string filename);
	/*!
	* \brief Resets the log level.
	* \param level The new log level.
	*/
	static void ResetLogLevel(LogLevel level);

	/*!
	* \brief C style formatted method for writing log messages. A message
	* is with the following format: [LEVEL] [TIME] message
	* \param level The log level of this message.
	* \param format The C format string.
	* \param ... Output items.
	* \return Returns a nonnegative integer on success,
	* or a negative number if error.
	*/
	static int Printf(LogLevel level, const char *format, ...);

private:
	static void CloseLogFile();
	// Returns current system time as a string.
	static std::string GetSystemTime();
	// Returns the string of a log level.
	static std::string GetLevelStr(LogLevel level);

	static LogLevel level_; // Only the message not less than level_ will be outputed.
	static std::FILE *file_; // A file pointer to the log file.
};