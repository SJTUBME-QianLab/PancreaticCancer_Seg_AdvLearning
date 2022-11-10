channel_size = 1
im_size = 256
cols = 256
rows = 256

Image_path = "/home/data/pancreas_cancer_%s_%s/T1/data/" % (im_size, im_size)
Label_path = "/home/data/pancreas_cancer_%s_%s/T1/label/" % (im_size, im_size)
Image_path_t2 = "/home/data/pancreas_cancer_%s_%s/T2/data/" % (im_size, im_size)
Label_path_t2 = "/home/data/pancreas_cancer_%s_%s/T2/label/" % (im_size, im_size)

MultiModal_T1_data_path = "/home/data/preprocessed/T1/data/"
MultiModal_T1_label_path = "/home/data/preprocessed/T1/label/"
MultiModal_T2_data_path = "/home/data/preprocessed/T2/data/"
MultiModal_T2_label_path = "/home/data/preprocessed/T2/label/"

Image_path_pancreas = "/home/data/pancreas_128_128/data/"
Label_path_pancreas = "/home/data/pancreas_128_128/label/"
all_to_npy_path = "/home/data/pancreas_cancer_all_to_256_256/"
all_to_128_path = "/home/data/all_to_128_128/"

T1_ap_list = [
    [
        "002345060",
        "002349703",
        "002338131",
        "002366693",
        "001657981",
        "002378100",
        "002371120",
        "002334623",
        "002395090",
        "002379897",
        "002379385",
        "002387315",
        "002405139",
        "002175490",
        "002396630",
        "002375282",
        "002339925",
        "002359725",
        "002371635",
        "002332907",
        "000270845",
        "002381519",
        "002398483",
        "002358189",
        "002386428",
        "002361206",
        "002329887",
        "002409810",
        "002363223",
        "001452384",
        "600469366",
        "002349189",
        "002389170",
        "002395276",
        "002342414",
        "600522234",
        "002388653",
        "002365184",
        "002363423",
        "002365108",
        "002365744",
        "002353553",
        "002380140",
        "002380188",
        "002372990",
        "002365003",
        "002374736",
        "002290034",
        "002360482",
        "600777276",
        "002322838",
        "002320307",
        "100068122",
        "002373487",
        "002342415",
        "002362356",
        "002399926",
        "002390703",
        "002356930",
        "002403102",
        "002346981",
        "001525966",
    ],
    [
        "002342414",
        "002371120",
        "600777276",
        "002372990",
        "002380188",
        "002399926",
        "002375282",
        "002398483",
        "002353553",
        "002403102",
        "002360482",
        "002380140",
        "002320307",
        "002345060",
        "002373487",
        "100068122",
        "002362356",
        "000270845",
        "002409810",
        "002405139",
        "002322838",
        "002363423",
        "001657981",
        "002396630",
        "002365003",
        "002365744",
        "002175490",
        "002386428",
        "002349189",
        "600469366",
        "002365108",
        "002359725",
        "002349703",
        "002395090",
        "002381519",
        "002365184",
        "002374736",
        "002390703",
        "002361206",
        "002338131",
        "002342415",
        "002379897",
        "002378100",
        "001452384",
        "002366693",
        "002387315",
        "002395276",
        "002379385",
        "002358189",
        "002371635",
        "002329887",
        "002356930",
        "002363223",
        "001525966",
        "600522234",
        "002290034",
        "002389170",
        "002388653",
        "002339925",
        "002334623",
        "002332907",
        "002346981",
    ],
    [
        "002345060",
        "002375282",
        "002363223",
        "002395276",
        "002365744",
        "002396630",
        "002175490",
        "002403102",
        "002342414",
        "002332907",
        "002349703",
        "001657981",
        "002379385",
        "002322838",
        "002380188",
        "002342415",
        "002349189",
        "002334623",
        "002358189",
        "002353553",
        "002365003",
        "002365108",
        "002387315",
        "002365184",
        "001525966",
        "002329887",
        "002360482",
        "002390703",
        "000270845",
        "002399926",
        "001452384",
        "002373487",
        "002362356",
        "002374736",
        "002338131",
        "002398483",
        "002389170",
        "002388653",
        "002346981",
        "002378100",
        "002371120",
        "002366693",
        "600522234",
        "002381519",
        "002379897",
        "002363423",
        "002409810",
        "002372990",
        "002405139",
        "002356930",
        "600469366",
        "600777276",
        "002320307",
        "002359725",
        "002371635",
        "002380140",
        "002395090",
        "002290034",
        "002386428",
        "002361206",
        "002339925",
        "100068122",
    ],
    [
        "002398483",
        "002373487",
        "002371635",
        "002387315",
        "002389170",
        "002396630",
        "002320307",
        "002366693",
        "001657981",
        "002395276",
        "002339925",
        "002362356",
        "002356930",
        "002290034",
        "600522234",
        "002395090",
        "002386428",
        "002342414",
        "002361206",
        "002329887",
        "002334623",
        "002374736",
        "002371120",
        "002175490",
        "002346981",
        "600777276",
        "002349189",
        "002365003",
        "002365184",
        "002363223",
        "002399926",
        "002409810",
        "002390703",
        "002338131",
        "002380140",
        "002358189",
        "002372990",
        "002353553",
        "002379385",
        "002365744",
        "002378100",
        "002359725",
        "001525966",
        "100068122",
        "002322838",
        "002405139",
        "002363423",
        "002349703",
        "600469366",
        "002360482",
        "002375282",
        "002332907",
        "002342415",
        "002403102",
        "002379897",
        "002380188",
        "002388653",
        "000270845",
        "002365108",
        "002381519",
        "001452384",
        "002345060",
    ],
    [
        "002398483",
        "001657981",
        "002365003",
        "002405139",
        "002290034",
        "001525966",
        "002362356",
        "002365184",
        "002358189",
        "002363223",
        "002322838",
        "002409810",
        "002399926",
        "600522234",
        "002342415",
        "002334623",
        "002360482",
        "002365744",
        "002371120",
        "002373487",
        "002175490",
        "002353553",
        "002372990",
        "002345060",
        "002371635",
        "100068122",
        "002356930",
        "002380188",
        "002390703",
        "000270845",
        "002386428",
        "002381519",
        "002379385",
        "002365108",
        "002374736",
        "002395276",
        "002361206",
        "002403102",
        "002346981",
        "002363423",
        "002320307",
        "002378100",
        "002349703",
        "002396630",
        "002395090",
        "002366693",
        "002349189",
        "002338131",
        "002359725",
        "002388653",
        "002380140",
        "002375282",
        "002389170",
        "002332907",
        "600777276",
        "600469366",
        "002379897",
        "002339925",
        "002387315",
        "002329887",
        "001452384",
        "002342414",
    ],
    [
        "002329887",
        "002395276",
        "002342415",
        "002334623",
        "002365108",
        "002365184",
        "000270845",
        "002380188",
        "002389170",
        "002349703",
        "002374736",
        "002378100",
        "002361206",
        "002371120",
        "002332907",
        "002362356",
        "002390703",
        "002363223",
        "002339925",
        "002363423",
        "002356930",
        "002396630",
        "002365003",
        "001657981",
        "002379897",
        "002380140",
        "002386428",
        "600777276",
        "001525966",
        "002373487",
        "002349189",
        "002371635",
        "002359725",
        "002379385",
        "002346981",
        "002399926",
        "600469366",
        "002365744",
        "002375282",
        "002409810",
        "002372990",
        "002360482",
        "002320307",
        "002358189",
        "002398483",
        "002175490",
        "002342414",
        "002290034",
        "002366693",
        "002387315",
        "002322838",
        "002388653",
        "001452384",
        "002395090",
        "002381519",
        "002403102",
        "100068122",
        "002405139",
        "002353553",
        "002338131",
        "002345060",
        "600522234",
    ],
    [
        "002371635",
        "002398483",
        "100068122",
        "002290034",
        "002361206",
        "002329887",
        "002362356",
        "001525966",
        "002175490",
        "002359725",
        "002374736",
        "002379385",
        "002339925",
        "002389170",
        "600777276",
        "002375282",
        "002396630",
        "002365744",
        "002338131",
        "002395090",
        "600469366",
        "002322838",
        "002320307",
        "002395276",
        "002356930",
        "002381519",
        "002372990",
        "600522234",
        "002409810",
        "002371120",
        "002342414",
        "002405139",
        "002332907",
        "002349703",
        "002399926",
        "002366693",
        "002334623",
        "002390703",
        "002345060",
        "002378100",
        "002386428",
        "002342415",
        "002346981",
        "002388653",
        "002373487",
        "002349189",
        "002403102",
        "000270845",
        "002360482",
        "002380140",
        "002353553",
        "002387315",
        "002363223",
        "002365003",
        "002379897",
        "002380188",
        "002365184",
        "002365108",
        "002358189",
        "002363423",
        "001452384",
        "001657981",
    ],
    [
        "002371635",
        "600777276",
        "002409810",
        "002379897",
        "002175490",
        "002381519",
        "002320307",
        "001452384",
        "002290034",
        "002374736",
        "002387315",
        "002363423",
        "002375282",
        "002389170",
        "002365184",
        "002358189",
        "002349703",
        "002373487",
        "002338131",
        "002346981",
        "002372990",
        "002386428",
        "001525966",
        "002361206",
        "002329887",
        "002366693",
        "002349189",
        "002360482",
        "002378100",
        "002388653",
        "002362356",
        "002342415",
        "002322838",
        "002365108",
        "000270845",
        "002395276",
        "001657981",
        "002363223",
        "002339925",
        "002353553",
        "002379385",
        "002380188",
        "002405139",
        "002345060",
        "002398483",
        "002390703",
        "002403102",
        "002380140",
        "002365744",
        "002334623",
        "002342414",
        "002359725",
        "002396630",
        "002356930",
        "002365003",
        "600469366",
        "002371120",
        "002332907",
        "002395090",
        "100068122",
        "002399926",
        "600522234",
    ],
    [
        "002320307",
        "002396630",
        "002390703",
        "002365003",
        "002342415",
        "002366693",
        "000270845",
        "002375282",
        "001525966",
        "002405139",
        "002378100",
        "002329887",
        "002381519",
        "002371635",
        "002363423",
        "002387315",
        "002360482",
        "002374736",
        "002380188",
        "002380140",
        "002365744",
        "600522234",
        "002342414",
        "002409810",
        "002290034",
        "002403102",
        "002353553",
        "100068122",
        "002372990",
        "002398483",
        "002399926",
        "002365108",
        "002332907",
        "002373487",
        "600777276",
        "002363223",
        "002346981",
        "002358189",
        "002395276",
        "002339925",
        "002334623",
        "002356930",
        "002349703",
        "001657981",
        "002386428",
        "002345060",
        "002389170",
        "002365184",
        "600469366",
        "002388653",
        "001452384",
        "002361206",
        "002349189",
        "002379897",
        "002359725",
        "002395090",
        "002322838",
        "002379385",
        "002362356",
        "002371120",
        "002338131",
        "002175490",
    ],
    [
        "001452384",
        "002372990",
        "002332907",
        "002346981",
        "002373487",
        "002175490",
        "002395276",
        "002388653",
        "002362356",
        "002342415",
        "002365184",
        "600469366",
        "002365744",
        "002386428",
        "002290034",
        "002396630",
        "002374736",
        "002375282",
        "002398483",
        "002360482",
        "002409810",
        "600522234",
        "002356930",
        "002361206",
        "002338131",
        "002381519",
        "002379385",
        "000270845",
        "002353553",
        "002371120",
        "002359725",
        "600777276",
        "002363223",
        "002365003",
        "002403102",
        "002389170",
        "001657981",
        "100068122",
        "002365108",
        "001525966",
        "002366693",
        "002378100",
        "002387315",
        "002345060",
        "002322838",
        "002380140",
        "002379897",
        "002342414",
        "002405139",
        "002339925",
        "002363423",
        "002395090",
        "002371635",
        "002320307",
        "002358189",
        "002349703",
        "002329887",
        "002399926",
        "002390703",
        "002380188",
        "002334623",
        "002349189",
    ],
    [
        "002349189",
        "002345060",
        "002409810",
        "002387315",
        "002362356",
        "001525966",
        "002334623",
        "002322838",
        "002358189",
        "000270845",
        "002339925",
        "002379385",
        "002388653",
        "002353553",
        "002360482",
        "002363423",
        "002398483",
        "002372990",
        "100068122",
        "002365108",
        "002395090",
        "002380140",
        "002399926",
        "002375282",
        "002175490",
        "600522234",
        "002395276",
        "002403102",
        "002365744",
        "002381519",
        "002320307",
        "001452384",
        "002290034",
        "002365003",
        "002366693",
        "002390703",
        "002371635",
        "600469366",
        "002356930",
        "600777276",
        "002380188",
        "002373487",
        "002361206",
        "002371120",
        "002359725",
        "002332907",
        "002405139",
        "002379897",
        "002389170",
        "002378100",
        "002346981",
        "002342415",
        "002338131",
        "002363223",
        "001657981",
        "002329887",
        "002374736",
        "002342414",
        "002386428",
        "002349703",
        "002396630",
        "002365184",
    ],
    [
        "002395090",
        "002374736",
        "002405139",
        "002366693",
        "002378100",
        "600469366",
        "002363223",
        "002346981",
        "002371120",
        "002380140",
        "002349189",
        "100068122",
        "002375282",
        "002399926",
        "002403102",
        "600522234",
        "002290034",
        "002372990",
        "002379385",
        "001525966",
        "002334623",
        "002332907",
        "002349703",
        "002362356",
        "002353553",
        "002338131",
        "002365184",
        "002386428",
        "002320307",
        "001452384",
        "002360482",
        "002359725",
        "002381519",
        "000270845",
        "002365108",
        "002388653",
        "002329887",
        "002345060",
        "002389170",
        "001657981",
        "002356930",
        "002371635",
        "002395276",
        "002322838",
        "002373487",
        "002365744",
        "002358189",
        "002396630",
        "002339925",
        "002390703",
        "002365003",
        "002342414",
        "002361206",
        "002398483",
        "002342415",
        "600777276",
        "002380188",
        "002175490",
        "002387315",
        "002379897",
        "002363423",
        "002409810",
    ],
    [
        "002398483",
        "600522234",
        "002372990",
        "002405139",
        "002361206",
        "002363223",
        "002366693",
        "000270845",
        "002349189",
        "002379897",
        "002290034",
        "002342414",
        "002409810",
        "002332907",
        "002365744",
        "002338131",
        "002349703",
        "002380140",
        "002381519",
        "002353553",
        "002358189",
        "001525966",
        "100068122",
        "002388653",
        "002360482",
        "002322838",
        "002380188",
        "002362356",
        "002378100",
        "002334623",
        "002363423",
        "002374736",
        "001657981",
        "002320307",
        "002390703",
        "002375282",
        "002365003",
        "002371120",
        "002379385",
        "002365184",
        "002399926",
        "002371635",
        "002373487",
        "002365108",
        "002386428",
        "002395090",
        "002346981",
        "002359725",
        "002175490",
        "001452384",
        "002395276",
        "002389170",
        "002387315",
        "002345060",
        "002342415",
        "002396630",
        "600777276",
        "002356930",
        "002339925",
        "002403102",
        "002329887",
        "600469366",
    ],
    [
        "002390703",
        "002320307",
        "002359725",
        "002332907",
        "600469366",
        "002365184",
        "002380188",
        "002387315",
        "002365108",
        "002398483",
        "002395276",
        "002175490",
        "002345060",
        "600777276",
        "002373487",
        "002365744",
        "002372990",
        "002342414",
        "002378100",
        "002353553",
        "002365003",
        "002363223",
        "002363423",
        "002349703",
        "002356930",
        "002366693",
        "002379897",
        "001657981",
        "600522234",
        "002290034",
        "002362356",
        "002395090",
        "002371635",
        "002371120",
        "002389170",
        "001525966",
        "002334623",
        "002346981",
        "002386428",
        "002405139",
        "100068122",
        "002361206",
        "002379385",
        "000270845",
        "002338131",
        "002339925",
        "002381519",
        "002409810",
        "002396630",
        "002322838",
        "002380140",
        "002388653",
        "002342415",
        "002358189",
        "002403102",
        "002360482",
        "002375282",
        "002374736",
        "002329887",
        "001452384",
        "002349189",
        "002399926",
    ],
    [
        "002381519",
        "002290034",
        "002395090",
        "600469366",
        "002338131",
        "002390703",
        "002379897",
        "002320307",
        "600777276",
        "002359725",
        "002405139",
        "002365108",
        "002374736",
        "002387315",
        "002360482",
        "002349189",
        "001525966",
        "002409810",
        "002389170",
        "002332907",
        "001452384",
        "002363223",
        "002396630",
        "002345060",
        "002346981",
        "002342415",
        "002334623",
        "002353553",
        "002388653",
        "002386428",
        "002365184",
        "002373487",
        "002380188",
        "002365003",
        "000270845",
        "002356930",
        "002380140",
        "002395276",
        "002349703",
        "002175490",
        "002363423",
        "002371635",
        "002322838",
        "002398483",
        "002329887",
        "100068122",
        "002375282",
        "002378100",
        "002366693",
        "002399926",
        "002372990",
        "002371120",
        "002361206",
        "002358189",
        "600522234",
        "002379385",
        "001657981",
        "002339925",
        "002342414",
        "002362356",
        "002403102",
        "002365744",
    ],
    [
        "002381519",
        "002365744",
        "002334623",
        "002372990",
        "002361206",
        "002339925",
        "002365184",
        "002405139",
        "002379897",
        "002409810",
        "002389170",
        "002360482",
        "002362356",
        "002346981",
        "001525966",
        "002366693",
        "002363223",
        "002365003",
        "002398483",
        "000270845",
        "002390703",
        "002356930",
        "002371635",
        "001452384",
        "600522234",
        "002332907",
        "600469366",
        "100068122",
        "002329887",
        "002375282",
        "002322838",
        "002386428",
        "002387315",
        "002395090",
        "002374736",
        "600777276",
        "002403102",
        "002342415",
        "002379385",
        "002345060",
        "002349703",
        "002363423",
        "002380188",
        "002373487",
        "002396630",
        "002320307",
        "002349189",
        "002365108",
        "002378100",
        "002371120",
        "002358189",
        "002395276",
        "002290034",
        "002175490",
        "002359725",
        "002388653",
        "002399926",
        "002353553",
        "001657981",
        "002380140",
        "002338131",
        "002342414",
    ],
    [
        "002363223",
        "002395276",
        "002353553",
        "002395090",
        "002378100",
        "002320307",
        "002361206",
        "002365744",
        "002365003",
        "002381519",
        "002175490",
        "100068122",
        "002342414",
        "002390703",
        "002375282",
        "002322838",
        "002362356",
        "002405139",
        "001452384",
        "002399926",
        "000270845",
        "002329887",
        "002379897",
        "002349189",
        "002403102",
        "001657981",
        "002388653",
        "600777276",
        "002360482",
        "001525966",
        "002389170",
        "002371635",
        "002346981",
        "002342415",
        "002349703",
        "600522234",
        "002379385",
        "002365184",
        "002372990",
        "002339925",
        "002409810",
        "002380140",
        "002366693",
        "002290034",
        "002332907",
        "002345060",
        "600469366",
        "002386428",
        "002380188",
        "002363423",
        "002373487",
        "002371120",
        "002334623",
        "002374736",
        "002387315",
        "002365108",
        "002338131",
        "002396630",
        "002359725",
        "002356930",
        "002358189",
        "002398483",
    ],
    [
        "002290034",
        "002395090",
        "002363423",
        "002372990",
        "002363223",
        "001525966",
        "002373487",
        "002387315",
        "002378100",
        "002381519",
        "002371635",
        "600469366",
        "002379385",
        "002380188",
        "002358189",
        "002405139",
        "002356930",
        "002390703",
        "002345060",
        "002409810",
        "001452384",
        "002365744",
        "002329887",
        "002334623",
        "002353553",
        "002361206",
        "002371120",
        "002374736",
        "000270845",
        "600777276",
        "002360482",
        "600522234",
        "002366693",
        "002396630",
        "002349703",
        "002349189",
        "002399926",
        "002380140",
        "002365184",
        "002389170",
        "002332907",
        "002365003",
        "100068122",
        "002386428",
        "002359725",
        "002346981",
        "002342415",
        "002365108",
        "002362356",
        "002375282",
        "002379897",
        "002403102",
        "002320307",
        "002175490",
        "002395276",
        "002339925",
        "001657981",
        "002322838",
        "002388653",
        "002398483",
        "002338131",
        "002342414",
    ],
    [
        "002379897",
        "002378100",
        "002374736",
        "002361206",
        "002363223",
        "002362356",
        "002399926",
        "002405139",
        "002320307",
        "002365184",
        "002365108",
        "002379385",
        "002403102",
        "002395276",
        "002363423",
        "002396630",
        "002395090",
        "002398483",
        "002380188",
        "002388653",
        "002349703",
        "600777276",
        "002359725",
        "000270845",
        "600469366",
        "002358189",
        "002386428",
        "002389170",
        "002373487",
        "002365003",
        "100068122",
        "001525966",
        "002390703",
        "002360482",
        "002322838",
        "002365744",
        "002290034",
        "002346981",
        "002371120",
        "002329887",
        "002356930",
        "002175490",
        "002334623",
        "002349189",
        "002381519",
        "002372990",
        "002366693",
        "002375282",
        "002371635",
        "001452384",
        "002387315",
        "002380140",
        "002339925",
        "002332907",
        "002353553",
        "002342414",
        "002409810",
        "600522234",
        "001657981",
        "002342415",
        "002338131",
        "002345060",
    ],
    [
        "002362356",
        "002375282",
        "002342415",
        "002365744",
        "002378100",
        "002372990",
        "002373487",
        "002409810",
        "002356930",
        "002349703",
        "002360482",
        "002380140",
        "002346981",
        "002405139",
        "002388653",
        "002329887",
        "002379897",
        "002175490",
        "002365003",
        "002363223",
        "002371120",
        "002332907",
        "002338131",
        "600522234",
        "002345060",
        "002366693",
        "002381519",
        "002399926",
        "002290034",
        "002365184",
        "001657981",
        "002353553",
        "002395276",
        "002358189",
        "002374736",
        "002398483",
        "002379385",
        "000270845",
        "100068122",
        "002403102",
        "002359725",
        "002361206",
        "002320307",
        "002390703",
        "002395090",
        "600469366",
        "002371635",
        "002396630",
        "600777276",
        "002380188",
        "002389170",
        "002342414",
        "002349189",
        "002387315",
        "002334623",
        "002365108",
        "002386428",
        "002363423",
        "001525966",
        "001452384",
        "002339925",
        "002322838",
    ],
]

T1_ADC_list = [
    "002396630",
    "000270845",
    "002349189",
    "002362356",
    "002378100",
    "002403102",
    "600522234",
    "002371120",
    "002358189",
    "002338131",
    "002322838",
    "002390703",
    "002409810",
    "002371635",
    "002363423",
    "002334623",
    "002365108",
    "002175490",
    "002379385",
    "002320307",
    "002395090",
    "002381519",
    "002342414",
    "002290034",
    "002375282",
    "002366693",
    "100068122",
    "002359725",
    "002387315",
    "002395276",
    "600469366",
    "002305286",
    "002399926",
    "002386428",
    "002380188",
    "002405139",
    "002398483",
    "002388653",
    "002353553",
    "002365003",
    "002329887",
    "002349703",
    "001452384",
    "002342415",
    "002374736",
    "002356930",
    "002361206",
    "002389170",
    "002372990",
    "001657981",
    "002306494",
    "002306542",
    "002379897",
    "002345060",
    "600417761",
    "002360482",
    "002373487",
    "002346981",
    "002363223",
    "002303226",
]

T1_DWI_list = [
    "002175490",
    "002388653",
    "600777276",
    "002373487",
    "002378100",
    "002320307",
    "002365108",
    "000270845",
    "002342414",
    "002338131",
    "002363423",
    "002349703",
    "002380188",
    "002342415",
    "002305286",
    "002386428",
    "002290034",
    "002365003",
    "002358189",
    "002395276",
    "002405139",
    "002365184",
    "002349189",
    "002306542",
    "002395090",
    "002403102",
    "002409810",
    "100068122",
    "002332907",
    "002390703",
    "002399926",
    "002322838",
    "002362356",
    "002360482",
    "002372990",
    "600522234",
    "002361206",
    "002381519",
    "002389170",
    "002379385",
    "001657981",
    "002353553",
    "002334623",
    "002398483",
    "002365744",
    "002371120",
    "002329887",
    "002363223",
    "002379897",
    "002345060",
    "002306494",
    "600469366",
    "600417761",
    "002380140",
    "002346981",
    "002374736",
    "002303226",
    "002359725",
    "001452384",
    "002356930",
    "002366693",
    "002387315",
    "002396630",
    "002375282",
]

T2_DWI_list = [
    "002320307",
    "002372990",
    "002305286",
    "002369440",
    "002338131",
    "002322838",
    "002398483",
    "002362356",
    "002366693",
    "002306602",
    "600459615",
    "000270845",
    "002396630",
    "100068122",
    "002345060",
    "002388653",
    "002358189",
    "002378100",
    "002403102",
    "002399926",
    "001452384",
    "002290034",
    "002346981",
    "002380140",
    "002365003",
    "002375282",
    "002306494",
    "002365184",
    "002390703",
    "002353553",
    "002360482",
    "002365744",
    "002306542",
    "002374736",
    "001657981",
    "002361206",
    "002409810",
    "002395090",
    "002349703",
    "002386428",
    "002373487",
    "002379385",
    "002381519",
    "002363423",
    "002342415",
    "002380188",
    "002371120",
    "600469366",
    "002365108",
    "002359725",
    "002395276",
    "002349189",
    "600522234",
    "002379897",
    "002387315",
    "002405139",
    "002303226",
    "002334623",
    "002175490",
    "002363223",
    "002348568",
    "002329887",
    "002342414",
    "600417761",
    "600777276",
    "002356930",
    "002332907",
    "002389170",
]

T2_ADC_list = [
    "002363423",
    "002379385",
    "002342415",
    "002409810",
    "002380188",
    "600469366",
    "002334623",
    "002363223",
    "002365108",
    "002405139",
    "002369440",
    "002365003",
    "002349189",
    "002379897",
    "002360482",
    "002290034",
    "002362356",
    "002378100",
    "002306602",
    "002396630",
    "002358189",
    "002372990",
    "002306494",
    "002373487",
    "002329887",
    "002386428",
    "002371120",
    "002338131",
    "002395090",
    "002388653",
    "600459615",
    "100068122",
    "002305286",
    "002399926",
    "002361206",
    "002381519",
    "002346981",
    "002374736",
    "002371635",
    "600417761",
    "002320307",
    "001657981",
    "002345060",
    "002398483",
    "002306542",
    "002387315",
    "002356930",
    "002389170",
    "002403102",
    "002353553",
    "600522234",
    "002359725",
    "001452384",
    "002366693",
    "002375282",
    "002348568",
    "002303226",
    "002349703",
    "002395276",
    "002390703",
    "002175490",
    "002342414",
    "002322838",
    "000270845",
]

ADC_DWI_list = [
    "002371120",
    "001657981",
    "002306494",
    "002398483",
    "002389170",
    "002306542",
    "600459615",
    "002373487",
    "002349189",
    "002369440",
    "002379897",
    "002390703",
    "002342414",
    "600522234",
    "002303226",
    "002380188",
    "002405139",
    "002374736",
    "002365108",
    "002361206",
    "002363223",
    "002305286",
    "002356930",
    "002381519",
    "002358189",
    "002362356",
    "002290034",
    "002363423",
    "002387315",
    "002360482",
    "002395276",
    "002353553",
    "002365003",
    "002329887",
    "002346981",
    "001452384",
    "002378100",
    "002306602",
    "002334623",
    "002372990",
    "002386428",
    "100068122",
    "002359725",
    "002395090",
    "002409810",
    "002379385",
    "002348568",
    "002349703",
    "600469366",
    "002396630",
    "002388653",
    "002322838",
    "002342415",
    "000270845",
    "002175490",
    "002338131",
    "002366693",
    "600417761",
    "002375282",
    "002320307",
    "002403102",
    "002345060",
    "002399926",
]

random_list_all = [
        22,
        66,
        38,
        1,
        23,
        18,
        46,
        43,
        7,
        64,
        17,
        20,
        4,
        3,
        0,
        29,
        42,
        40,
        8,
        65,
        48,
        2,
        5,
        50,
        39,
        32,
        44,
        34,
        15,
        27,
        56,
        31,
        9,
        57,
        63,
        14,
        12,
        62,
        51,
        49,
        28,
        60,
        61,
        30,
        35,
        59,
        45,
        41,
        58,
        52,
        21,
        10,
        26,
        24,
        25,
        53,
        6,
        13,
        54,
        47,
        33,
        19,
        16,
        36,
        37,
        11,
        55,
    ]

