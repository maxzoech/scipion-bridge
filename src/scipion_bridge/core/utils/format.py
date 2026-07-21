

def format_list(items, oxford_comma=True):
    if not items:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        # We generally don't use a comma for only two items
        return f"{items[0]} and {items[1]}"
    
    # 3 or more items
    comma_str = "," if oxford_comma else ""
    return f"{', '.join(items[:-1])}{comma_str} and {items[-1]}"