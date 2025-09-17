# This file demonstrates the behaviour of calculator module
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): 
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
def power(a, b): return a ** b


def calc(a, b, operation='add'):
    if operation == 'add':
        return add(a, b)
    elif operation == 'subtract':
        return subtract(a, b)
    elif operation == 'multiply':
        return multiply(a, b)
    elif operation == 'divide':
        return divide(a, b)
    elif operation == 'power':
        return power(a, b)
    else:
        raise ValueError("Invalid operation")
    
