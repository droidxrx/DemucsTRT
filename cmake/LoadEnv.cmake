#[=======================================================================[.rst:
LoadEnv
-------

A module to load environment variables from a ``.env`` file into the CMake
project. This script parses the file natively and sets each variable in
both the calling scope and the environment scope.

.. command:: load_env_file

  .. code-block:: cmake

    load_env_file(<file_path> [REQUIRED])

  Parses the specified ``.env`` file and exports the key-value pairs.

  ``<file_path>``
    Path to the ``.env`` file.

  ``REQUIRED``
    If provided, halts configuration with a FATAL_ERROR if the file is
    not found. If not provided, a WARNING is issued and the function returns.

Features:
  * Pure native CMake parsing (no external scripts like Python or Bash).
  * Ignores empty lines and full-line comments (starting with ``#``).
  * Trims leading/trailing whitespaces around keys and values.
  * Strips surrounding single (``'``) or double (``"``) quotes.
  * Preserves ``#`` characters inside quoted strings.
  * Correctly discards inline comments outside of quotes.
  * Safe against string values containing semicolons (``;``).
#]=======================================================================]

cmake_minimum_required(VERSION 3.15)

function(load_env_file env_file)
    # Parse function arguments
    cmake_parse_arguments(PARSE_ARGV 1 ARG "REQUIRED" "" "")

    # Check file existence and handle errors based on the REQUIRED flag
    if(NOT EXISTS "${env_file}")
        if(ARG_REQUIRED)
            message(FATAL_ERROR "load_env_file: Required environment file not found: ${env_file}")
        else()
            message(WARNING "load_env_file: Environment file not found: ${env_file}")
            return()
        endif()
    endif()

    # Read the entire file
    file(READ "${env_file}" file_content)

    # Protect semicolons from being evaluated as CMake list separators during the split
    set(semicolon_placeholder "[[__CMAKE_ENV_SEMICOLON_PLACEHOLDER__]]")
    string(REPLACE ";" "${semicolon_placeholder}" file_content "${file_content}")

    # Normalize carriage returns and split into a list of lines
    string(REPLACE "\r\n" "\n" file_content "${file_content}")
    string(REPLACE "\r" "\n" file_content "${file_content}")
    string(REPLACE "\n" ";" env_lines "${file_content}")

    # Process each line
    foreach(line IN LISTS env_lines)
        # Restore semicolons for this specific line
        string(REPLACE "${semicolon_placeholder}" ";" line "${line}")

        # Trim whitespace
        string(STRIP "${line}" trimmed_line)

        # Ignore empty lines and full-line comments
        if(trimmed_line STREQUAL "" OR trimmed_line MATCHES "^#")
            continue()
        endif()

        # Find the first '=' character
        string(FIND "${line}" "=" equal_pos)
        if(equal_pos EQUAL -1)
            continue() # Not a valid key-value pair, skip line
        endif()

        # Extract Key and Value strings
        string(SUBSTRING "${line}" 0 ${equal_pos} key)
        math(EXPR val_pos "${equal_pos} + 1")
        string(SUBSTRING "${line}" ${val_pos} -1 value)

        # Trim spaces around the key and value
        string(STRIP "${key}" key)
        string(STRIP "${value}" value)

        # ---------------------------------------------------------
        # Value Scanner: Handle quotes, escapes, and inline comments
        # ---------------------------------------------------------
        string(LENGTH "${value}" val_len)
        set(i 0)
        set(final_val "")
        set(in_quotes FALSE)
        set(quote_char "")
        set(escaped FALSE)

        # Detect opening quotes
        if(val_len GREATER 0)
            string(SUBSTRING "${value}" 0 1 first_char)
            if(first_char STREQUAL "\"" OR first_char STREQUAL "'")
                set(in_quotes TRUE)
                set(quote_char "${first_char}")
                set(i 1) # Skip the opening quote
            endif()
        endif()

        while(i LESS val_len)
            string(SUBSTRING "${value}" ${i} 1 char)

            if(escaped)
                string(APPEND final_val "${char}")
                set(escaped FALSE)
            elseif(char STREQUAL "\\")
                set(escaped TRUE)
            elseif(in_quotes)
                if(char STREQUAL quote_char)
                    # Closing quote found, stop processing (ignores trailing inline comments)
                    set(in_quotes FALSE)
                    break()
                else()
                    string(APPEND final_val "${char}")
                endif()
            else()
                if(char STREQUAL "#")
                    # Start of unquoted inline comment, stop processing
                    break()
                else()
                    string(APPEND final_val "${char}")
                endif()
            endif()

            math(EXPR i "${i} + 1")
        endwhile()

        # Final trim for trailing spaces on unquoted strings
        string(STRIP "${final_val}" final_val)

        # Set the parsed variable in the parent scope and the environment
        set("${key}" "${final_val}" PARENT_SCOPE)
        set(ENV{${key}} "${final_val}")

    endforeach()
endfunction()