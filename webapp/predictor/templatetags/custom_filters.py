from django import template

register = template.Library()

@register.filter(name='multiply')
def multiply(value, arg):
    """Multiplies the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter(name='replace')
def replace(value, arg):
    """
    Replaces characters in a string.
    Usage: {{ value|replace:"_, " }} -> replaces '_' with ' '
    """
    try:
        if len(arg.split(',')) == 2:
            old, new = arg.split(',')
            return str(value).replace(old, new)
        return str(value).replace(arg, '')
    except (ValueError, TypeError):
        return value
