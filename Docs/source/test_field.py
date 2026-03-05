def field(apply_func=False, raw=False):
    """Return the contents of the bibliography entry field."""
    result = "start"
    try:
        # print("try: start")
        if not raw:
            result += ", not raw"
    except KeyError:
        print("KeyError")
    # finally:
    #     print("")
    else:
        result += ", else"
        if apply_func:
            result += ", apply_func"
        return result

for apply_func in [True, False]:
    for raw  in [True, False]:
        result = field(apply_func=apply_func, raw=raw)
        print(f"\nfield(apply_func={apply_func}, raw={raw}) = {result}")
