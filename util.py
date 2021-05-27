def request_xml(page_number):
    encodingKey = "Ud3EhGOpbJ2xenIcXMsJcMJhXzH8U8v29DapZ0PioWkUFZmz1T8W4WJ7cjNlEG9yXPZGsFENb9jGqUBLAzlpjg%3D%3D"
    decodingKey = "Ud3EhGOpbJ2xenIcXMsJcMJhXzH8U8v29DapZ0PioWkUFZmz1T8W4WJ7cjNlEG9yXPZGsFENb9jGqUBLAzlpjg=="

    url = "http://apis.data.go.kr/1390804/NihhsMushroomImageInfo/selectMushroomImageList?ServiceKey=" + encodingKey + "&pageIndex="+str(page_number)
    request = Request(url)
    response = urlopen(request)
    rescode = response.getcode()

    if(rescode==200):
        response_body = response.read()
        dom = minidom.parseString(response_body.decode('utf-8'))
        xml_str = dom.toprettyxml()
    else:
        print("Error Code:" + rescode)

    return xml_str

def save_pickle(save_file,file_name):
    with open(file_name,"wb") as f:
        pickle.dump(save_file, f)
        
def open_pickle(file_name):
    with open(file_name,"rb") as f:
        data = pickle.load(f)
    return data