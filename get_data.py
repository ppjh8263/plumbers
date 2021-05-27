from urllib import request
from xml.etree.ElementTree import fromstring
import numpy as np
import pandas as pd
from PIL import Image

import util


page_number=1
scNameKr=[]
imgFileUrl=[]
while(True):
    xml_parsing=fromstring(util.request_xml(page_number))
    if xml_parsing.find('./root/resultCnt').text!='0':
        for i in xml_parsing.findall('./root/result'):
            scNameKr.append(i.find('./scNameKr').text)
            imgFileUrl.append(i.find('./imgFileUrl').text)
        page_number=page_number+1
    else:
        break    
        
result={'name':scNameKr,'url':imgFileUrl}

file_name='name_url.pickle'
util.save_pickle(result,'name_url.pickle')

images=[]
fig_path='./images/'
for i in result['url']:
    file_name=fig_path+i.split('/')[-1]
    request.urlretrieve(i, file_name)
    images.append(np.array(Image.open(file_name)))


#피클파일 저장
result['images']=images
util.save_pickle(images,"images.pickle")
util.save_pickle(result,"name_url_image.pickle")


result_pickle=util.open_pickle('name_url_image.pickle')

df_result = pd.DataFrame(result_pickle)
print(df_result['name'].unique())