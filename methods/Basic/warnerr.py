def warnerr(*args):
    """
    Function to output warning or error messages.

    Parameters:
        *args: Variable-length argument list used to receive any number of input parameters.

    """
    global ERR_NOT_WARN

    # Check whether the global variable ERR_NOT_WARN is empty or evaluates to False
    if ERR_NOT_WARN is None or not ERR_NOT_WARN:
        # If ERR_NOT_WARN is empty or evaluates to False, call the warnmsg function to output the warning message
        warnmsg(*args)
    else:
        # Otherwise, call the error function to output the error message and terminate the program execution
        error(*args)

def warnmsg(*args):
    """
    Function to output warning messages.

    Parameters:
        *args: Variable-length argument list used to receive any number of input parameters.

    """
    # Here, we can define the operation to output warning messages; here, we simply print them
    print("Warning message:", *args)

def error(*args):
    """
    Function to output error messages and terminate program execution.

    Parameters:
        *args: Variable-length argument list used to receive any number of input parameters.

    """
    # Here, we can define the operation to output error messages and terminate program execution; here, we simply print them
    print("Error message:", *args)
    # Terminate program execution
    raise SystemExit

# Test code
ERR_NOT_WARN = True
warnerr("This is a warning")
ERR_NOT_WARN = False
warnerr("This is a warning")
warnerr("This is a warning")