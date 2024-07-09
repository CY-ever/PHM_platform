from flask import abort


def str_to_float(x_str):
    print("x_str", x_str, type(x_str))
    try:
        if "," in x_str:
            x = x_str.split(",")
            print(x)
            for i in range(len(x)):
                x[i] = float(x[i])
            print(x)
            x = tuple(x)
            # print("拆分结果为:", x)
            return x
        else:
            print("x_str", x_str)
            x = (float(x_str),)
            return x
    except:
        abort(400,
              "Illegal input format. Please use ',' to separate numbers, and please do not contain other characters.")


def str_to_int(x_str):
    print("x_str", x_str, type(x_str))
    try:
        if "," in x_str:
            x = x_str.split(",")
            print(x)
            for i in range(len(x)):
                x[i] = int(x[i])
            print(x)
            x = tuple(x)
            # print("拆分结果为:", x)
            return x
        else:
            print("**************")
            x = (int(x_str),)
            return x
    except:
        abort(400,
              "Illegal input format. Please use ',' to separate numbers, and please do not contain other characters and float.")
