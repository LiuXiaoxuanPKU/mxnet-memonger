

c_time_cnt = {}
opt_time_cnt = {}
if __name__ == "__main__":
    start_c = False
    start_opt = False


    with open("./out", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line == "MXNET_C_API":
                start_c = True
            elif line == "operator":
                start_c = False
                start_opt = True

            if start_c:
                tokens = line.split(" ")
                if tokens[0] == "=================" or tokens[0] == "Name":
                    continue
                c_time_cnt[tokens[0]] = float(tokens[2])

            if start_opt:
                tokens = line.split(" ")
                if tokens[0] == "=================" or tokens[0] == "Name":
                    continue
                opt_time_cnt[tokens[0]] = float(tokens[2])

    total_c_time = 0
    total_opt_time = 0

    for key in c_time_cnt.keys():
        total_c_time += c_time_cnt[key]

    for key in opt_time_cnt.keys():
        total_opt_time += opt_time_cnt[key]

    print("Total C time: ", total_c_time)
    print({k: v for k, v in sorted(c_time_cnt.items(), key=lambda item: item[1])})
    print("Total operator time: ", total_opt_time)
    print({k: v for k, v in sorted(opt_time_cnt.items(), key=lambda item: item[1])})



