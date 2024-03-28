import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import html

titles = []
isFirst = True
head_url = 'https://www.transfermarkt.world'
url = '/spieler-statistik/wertvollstespieler/marktwertetop?land_id=0&ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&kontinent_id=0&plus=1'
while (isFirst or titles):
    cookie = "TMSESSID=44020d2b27eeeb04d68bba255e538ca1; _ga=GA1.2.604857484.1695144762; _gid=GA1.2.456467293.1695144762; _sp_v1_ss=1:H4sIAAAAAAAAAItWqo5RKimOUbKKxsrIAzEMamN1YpRSQcy80pwcILsErKC6lgwJpVgAEA5-UnQAAAA%3D; _sp_v1_p=521; _sp_v1_data=656982; euconsent-v2=CPyWO0APyWO0AAGABCENASEsAP_AAEPAAAwIIJoF9CpETWFAAW59AJsEAAQXwVBhJmAgAgCAACABABAAYAwEkGAAIASAAAAAAAAAIBIBAAAAAAEAAAAAYIgAABEIAgAAoAAIIAAAEAAAAAAAAAAIAggAAAAYAAABAAAAiACAAAIAQEAAAAAAAAAAAIAAAAABAAAAAAAAAAAAAAAAAAAAggnAOAAQAA4ANAAigBHADugIOAhABdQDtgH2AP-Ap8BggEEwDwkBcABAACwAKgAcABAADIAGgARAAmABPAD8AIQARwAmgBSgDDAHdAPwA_QDFAIdAUeAvMBkgUACAIoIAEABIAEcAjgBOw0AEBDo6A0AAsACoAHAAQAAyABoAEQAJgATwAxAB-AE0AKUAYcA_AD9AIsAR0AxQB1AD7AIdAReAo8BeYDJAGWDwAIAihwAcAC4AJAAjgBQAEcARwAnYiACAh0QABABIQgEAALACYAGIARwApQB3AGKAOoQAAgEcJQDwAEAALAA4AEQAJgAYgBHAD8AMUAdQBDoCLwFHgLzAZISACAAXACOARwBlhSAuAAsACoAHAAQAA0ACIAEwAJ4AUgAxAB-AFLAPwA_QCLAEdAMUAdQA-wCHQEXgLzAZIAywoAFAAuACQAGQARwCDgA.YAAAAAAAAAAA; consentUUID=4f7f26f6-46dc-4b01-b4ed-e26f34bfa360_23; ioam2018=000d3cfe8b9ee9a146509db4b%3A1725471180343%3A1695144780342%3A.transfermarkt.world%3A5%3Atransfer%3Aausland_rest_r%3Anoevent%3A1695153807040%3A3xzp3j"
    response = requests.get(head_url + url, headers={"Cookie": cookie, "Accept-Encoding": "gzip, deflate, br",
                                                     "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}).content

    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', class_='items')

    # Extract data from the table
    data = []
    isFirstRow = True
    for row in table.find_all('tr'):
        cols = row.find_all(['th', 'td'])
        if isFirstRow:
            cols = [col.div.get('title') if col.div else col.text.strip() for col in cols]
        else:
            cols = [col.img.get('title') if col.img else col.text.strip() for col in cols]
        if isFirstRow:
            cols.insert(2, 'Амплуа')
            cols.append('Гол+Пас за один матч')
            isFirstRow = False
        else:
            if (len(cols) > 2):
                cols.pop(1)
                cols.pop(1)
                if int(cols[7]) != 0:
                    val = (int(cols[8]) + int(cols[10])) / int(cols[7])
                    cols.append(float('{:.2f}'.format(val)))

        data.append(cols)

    # pandas from ar to dataframe
    df = pd.DataFrame(data[1:], columns=data[0])
    df_sorted = df.sort_values(by='#')
    if isFirst:
        df_sorted.head(25).to_csv('output.csv', index=False)
        isFirst = False
    else:
        df_sorted.head(25).to_csv('output.csv', mode='a', header=False, index=False)

    tree = html.fromstring(response)

    # Extract data using XPath
    titles = tree.xpath('//li[@class="tm-pagination__list-item tm-pagination__list-item--icon-next-page"]/a/@href')
    if titles:
        url = titles[0]
    print(titles)
