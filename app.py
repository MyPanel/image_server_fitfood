from flask import Flask, request, send_file, jsonify, session, escape, Response
import food_keras as food
import build_model as build
# from werkzeug import secure_filename
from PIL import Image
import numpy as np
import keras.backend.tensorflow_backend as tb
import os

import csv
from bs4 import BeautifulSoup
import requests

from flask_cors import CORS

import mysql.connector

import random

import bcrypt 

from itertools import combinations

from io import StringIO, BytesIO
import base64

import threading

tb._SYMBOLIC_SCOPE.value = True

# config = {'user': 'root',
#         'password': '',
#         'host': 'localhost', 
#         'db': 'fitfood_v52'}
        
config = {'user': 'wdj2-user',
            'password': 'wdj2-Fitfood', 
            'host': 'ec2-34-239-220-61.compute-1.amazonaws.com', 
            'db': 'fitfood_v5'}
conn = mysql.connector.connect(**config)
cur = conn.cursor(buffered=True)

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

CORS(app, supports_credentials=True)

image_size = 64

def get_image(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img

@app.route('/test', methods=['POST'])
def getCal():
    # tb._SYMBOLIC_SCOPE.value = True

    #아래 세줄 base64로 이미지 받을때
    str_img = request.form['img']
    image = base64.b64decode(str_img)
    img = Image.open(BytesIO(image)) 

    resNum = request.form['resNum']

    #아래 두줄 파일로 이미지 받을 때
    # str_img = request.files['img']
    # img = Image.open(str_img)

    # print(str_img)
    tb._SYMBOLIC_SCOPE.value = True 

    caltech_dir = './images2/' + str(resNum)
    categories = os.listdir(caltech_dir)
    if ".DS_Store" in categories :
           categories.remove('.DS_Store')
    categories = sorted(categories)
    image_size = 64
    X = []
    files = []

    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(in_data)
    X = np.array(X)

    model = food.build_model(X.shape[1:], resNum)
    model.load_weights("./models/" + str(resNum) + ".hdf5")
    pre = model.predict(X)
    for i, p in enumerate(pre):
        y = p.argmax()

    # print(categories)
    # print(y)
    # query = "select food_id from foods where store_id = \'"+str(resNum)+"\' and food_name = \'"+str(categories[y])+"\'"
    # print(query)

    # cur.execute(query)
    # data = cur.fetchall()
    # print(data)
    # foodId = data[0][0]
    # foodId = cur.fetchall()[0][0]
    
    # print(foodId)

    # return str(foodId)

    return str(categories[y])

@app.route('/recommend/<user_id>', methods=['POST'])
def getMealLists(user_id):
    formData = request.form
    mealKindList = str(formData['kind']).split(',')
    # mealKindList = [0, 1, 1]
    # mealKindList = [0, 1]
    # mealKindList = [0]
    # mealKindList = [1]
    for i in range(len(mealKindList)):
        mealKindList[i] = int(mealKindList[i])
    todayEatNum = len(mealKindList)

    # 0 : store , 1 : recipe

    # get foodNutrient Info
    cur.execute('select nutrient_carbohydrate, nutrient_protein, nutrient_fat, nutrients.food_id, food_name  from foods join nutrients  on foods.food_id = nutrients.food_id')
    foodList = cur.fetchall()

    foodIdList = [ i for i in range(len(foodList))]
    
    # cur.execute('select food_id  from nutrients where food_id is not null')
    # foodIdList = [i[0] for i in cur.fetchall()]
    
    # get Recipe
    # fw = open('recipe3.csv', 'r')
    # recipeRdr = csv.reader(fw)
    # recipeList = [ line for line in recipeRdr ]

    # get recipe info
    cur.execute('select nutrient_carbohydrate, nutrient_protein, nutrient_fat, nutrients.recipe_id, recipe_food, recipe_image_2 from recipes join nutrients  on recipes.recipe_id = nutrients.recipe_id')
    recipeList = cur.fetchall()

    recipeIdList = [ -(i+1) for i in range(len(recipeList)) ]

    # cur.execute('select recipe_id from nutrients where recipe_id is not null')
    # recipeIdList =[ -i[0] for i in cur.fetchall() ]

    AllMealList = []
    storeMealList = []
    recipeMealList = []

    if len(set(tuple(mealKindList))) == 1:
        if mealKindList[0] == 0:
            AllMealList = combinations(foodIdList, todayEatNum)
        elif mealKindList[0] == 1:  
            AllMealList = combinations(recipeIdList, todayEatNum)
    elif len(mealKindList) == 3:
        storekindChekNum = 0
        recipekindChekNum = 0
        for i in mealKindList:
            if i == 0:
                storekindChekNum += 1
            else:
                recipekindChekNum += 1
        if storekindChekNum == 2:
            storeMealList = list(combinations(foodIdList, 2))
            for i in recipeIdList:
                # print(i)
                for j in storeMealList:
                    # print(i)
                    todayMealList = []
                    index = 0
                    for k in mealKindList:
                        if k == 0:
                            todayMealList.append(j[index])
                            index += 1
                        else:
                            todayMealList.append(i)
                    AllMealList.append(todayMealList)
        elif recipekindChekNum == 2:
            recipeMealList = list(combinations(recipeIdList, 2))
            for i in foodIdList:
                for j in recipeMealList:
                    todayMealList = []
                    index = 0
                    for k in mealKindList:
                        if k == 1:
                            todayMealList.append(j[index])
                            index += 1
                        else:
                            todayMealList.append(i)
                    AllMealList.append(todayMealList)
    else:
        for i in foodIdList:
            for j in recipeIdList:
                AllMealList.append([i, j])

    # get eatten meal info
    if todayEatNum != 3:
        usersEaten_select_query = select_str('foodeatens',
                                            ['nutrient_id'],
                                            {'user_id': user_id})[:-1]                  
        usersEaten_select_query += "and DATE_FORMAT(eaten_start, \"%Y-%m-%d\") = CURDATE()"
        # print(usersEaten_select_query)
        cur.execute(usersEaten_select_query)
        eatenList = cur.fetchall()
        # eatenList = [[1]]
        # print(eatenList)
    
    user_select_query = select_str('users',
                            ['user_height'],
                            {'user_id': user_id})
    cur.execute(user_select_query)

    userInfo = cur.fetchall()[0]
    rightCal = (userInfo[0] - 100) * 0.9 * 32
    rightCal = (170 - 100) * 0.9 * 25
    # 탄수화물 55~65% 4
    # 단백질 7~20% 4
    # 지방 15~30% 9 

    goodTan = [rightCal/4 * 0.55, rightCal/4 * 0.65]
    goodDan = [rightCal/4 * 0.07, rightCal/4 * 0.2]
    goodJi = [rightCal/9 * 0.15, rightCal/9 * 0.3]

    foodeaten_select_query = 'select nutrient_carbohydrate, nutrient_protein, nutrient_fat  from foodeatens join nutrients  on foodeatens.food_id = nutrients.food_id where user_id = 1 and eaten_start > date_add(now(),interval -7 day)'
    cur.execute(foodeaten_select_query)
    data = cur.fetchall()
    # print(len(data))
    if len(data) > 0:
        tan = 0
        dan = 0
        ji = 0
        for i in data:
            tan += i[0]
            dan += i[1]
            ji += i[2]

        # print(goodTan[0])
        # print(tan / (len(data)/3))
        # print(goodTan[1])
        # print('==============')
        # print(goodDan[0])
        # print(dan / (len(data)/3))
        # print(goodDan[1])
        # print('==============')
        # print(goodJi[0])
        # print(ji / (len(data)/3))
        # print(goodJi[1])

        # 탄수화물 55~65% 4 tan / (len(data)/3)
        # 단백질 7~20% 4 dan / (len(data)/3)
        # 지방 15~30% 9 ji / (len(data)/3)
        eatnutrients = (tan / (len(data)/3) * 4) + (dan / (len(data)/3) * 4) + (ji / (len(data)/3) * 9)
        # tanFailChecker = 0.55 < (tan / (len(data)/3) * 4) / eatnutrients
        tanFailChecker = -1 if (tan / (len(data)/3) * 4) / eatnutrients < 0.55 else (1 if 0.65 < (tan / (len(data)/3) * 4) / eatnutrients else 0) 
        danFailChecker = -1 if (dan / (len(data)/3) * 4) / eatnutrients < 0.07 else (1 if 0.2 < (dan / (len(data)/3) * 4) / eatnutrients else 0) 
        jiFailChecker = -1 if (ji / (len(data)/3) * 9) / eatnutrients < 0.15 else (1 if 0.3 < (ji / (len(data)/3) * 9) / eatnutrients else 0) 
        print((tan / (len(data)/3) * 4) / eatnutrients )
        print( (dan / (len(data)/3) * 4) / eatnutrients)
        print((ji / (len(data)/3) * 9) / eatnutrients)
        # tanFailChecker = -1 if tan < goodTan[0] / (len(data)/3) else (1 if  goodTan[1] < tan / (len(data)/3) else 0) 
        # danFailChecker = -1 if dan < goodDan[0] / (len(data)/3) else (1 if  goodDan[1] < dan / (len(data)/3) else 0) 
        # jiFailChecker = -1 if ji < goodJi[0] / (len(data)/3) else (1 if  goodJi[1] < ji / (len(data)/3) else 0) 
        FailCheckes = [ tanFailChecker, danFailChecker, jiFailChecker]
    else:
        FailCheckes = [ 0, 0, 0 ]

    for i in range(3 - todayEatNum):
        print(str(eatenList[i][0]))
        nutrient_select_query = select_str('nutrients',
                                        ['nutrient_carbohydrate', 'nutrient_protein', 'nutrient_fat'],
                                        {'nutrient_id': str(eatenList[i][0])})
        cur.execute(nutrient_select_query)
        eattedNutrientInfo = cur.fetchall()[0]
        print(eattedNutrientInfo[0])
        goodTan[0] -= float(eattedNutrientInfo[0])
        goodTan[1] -= float(eattedNutrientInfo[0])
        goodDan[0] -= float(eattedNutrientInfo[1])
        goodDan[1] -= float(eattedNutrientInfo[1])
        goodJi[0] -= float(eattedNutrientInfo[2])
        goodJi[1] -= float(eattedNutrientInfo[2])

    RightMealList = []
    testMealList1 = []
    testMealList2 = []

    for i in AllMealList:
        tan = 0
        dan = 0
        ji = 0
        foodIdList = []
        foodNameList = []
        foodImgList = []
        for j in range(todayEatNum):
            # print(i[j])
            if i[j] >= 0:
                tan += foodList[i[j]][0]
                dan += foodList[i[j]][1]
                ji += foodList[i[j]][2]
                foodId = foodList[i[j]][3]
                foodName = foodList[i[j]][4]
                foodIdList.append(foodId)
                foodNameList.append(foodName)
                foodImgList.append(-1)
                # for k in foodList:
                #     if k[3] == i[j]:
            else:
                # if -i[j] - 1 < 1000:
                    # tan += float(recipeList[-i[j] - 1][6])
                    # dan += float(recipeList[-i[j] - 1][7])
                    # ji += float(recipeList[-i[j] - 1][8])
                    # foodNameList.append(recipeList[-i[j] - 1][1])
                    # foodImgList.append(recipeList[-i[j] -1][11])
                tan += recipeList[-i[j] - 1][0]
                dan += recipeList[-i[j] - 1][1]
                ji += recipeList[-i[j] - 1][2]
                foodIdList.append(-int(recipeList[-i[j] - 1][3]))
                foodNameList.append(recipeList[-i[j] - 1][4])
                foodImgList.append(recipeList[-i[j] - 1][5])
                # else:
                #     break
        # foodIdList = i 
        eatSmallcheckTan = tan - goodTan[0]
        eatBigcheckTan = tan - goodTan[1]
        eatSmallcheckDan = dan - goodDan[0]
        eatBigcheckDan = dan - goodDan[1] 
        eatSmallcheckJi = ji - goodJi[0]
        eatBigcheckJi = ji - goodJi[1]
        failNutrientScores = []
        if eatSmallcheckTan > 0 and eatBigcheckTan < 0:
            if eatSmallcheckDan > 0 and eatBigcheckDan < 0:
                if eatSmallcheckJi > 0 and eatBigcheckJi < 0:
                    if 329 in foodIdList:
                        testMealList1.append([foodIdList, foodNameList, foodImgList])
                    if '13' in foodIdList:
                        testMealList2.append([foodIdList, foodNameList, foodImgList])
                    if -1 in FailCheckes or 1 in FailCheckes:
                        nutrientBigSmallChecks = [[eatSmallcheckTan, eatBigcheckTan], [eatSmallcheckDan, eatBigcheckDan ], [eatSmallcheckJi, eatBigcheckJi] ]
                        for index in range(len(FailCheckes)):
                            if FailCheckes[index] == -1:
                                failNutrientScores.append(nutrientBigSmallChecks[index][0])
                            elif FailCheckes[index] == 1:
                                failNutrientScores.append(nutrientBigSmallChecks[index][1])
                            else:
                                failNutrientScores.append(0)
                        failscore = 0 
                        FailScore = failNutrientScores[0] * 4 + failNutrientScores[1] * 4 + failNutrientScores[2] * 9
                        FailAbsScore = abs((failNutrientScores[0] * failNutrientScores[0] * 4 * 4 + failNutrientScores[1] * failNutrientScores[1] * 4 * 4 + failNutrientScores[2] * failNutrientScores[2] * 9 * 9) / (16 * 16 * 81))
                        RightMealList.append([foodIdList, foodNameList, foodImgList, FailAbsScore])
                    else:
                        RightMealList.append([foodIdList, foodNameList, foodImgList])

    randomMeal = []
    resultIdList= []
    # print(RightMealList)
    print('======================')

    if -1 in FailCheckes or 1 in FailCheckes:
        RightMealList = sorted(RightMealList, key = lambda x : x[3])
        equalCheckList = []
        index = 0
        loopChecker = True
        while loopChecker:
            equalChecker = False
            if index == 0:
                equalCheckList.append(RightMealList[index][1])
                randomMeal.append(RightMealList[index])
            else:
                for i in equalCheckList:
                    for k in RightMealList[index][1]:
                        if k in i:
                            equalChecker = False
                            break
                        else:
                            equalChecker = True
                if equalChecker:
                    equalCheckList.append(RightMealList[index][1])
                    randomMeal.append(RightMealList[index])
            if len(randomMeal) == 5 or len(RightMealList) == index:
                loopChecker = False
            index += 1
    else:
        equalCheckList = []
        index = 0
        loopChecker = True
        while loopChecker:
            equalChecker = False
            randomNum = random.randrange(0, len(RightMealList))
            if index == 0:
                equalCheckList.append(RightMealList[randomNum][1])
                randomMeal.append(RightMealList[randomNum])
            else:
                for i in equalCheckList:
                    for k in RightMealList[randomNum][1]:
                        if k in i:
                            equalChecker = False
                            break
                        else:
                            equalChecker = True
                if equalChecker:
                    equalCheckList.append(RightMealList[randomNum][1])
                    randomMeal.append(RightMealList[randomNum])
            if len(randomMeal) == 5 or len(RightMealList) == 10000:
                loopChecker = False
            index += 1
        # for i in range(5):
        #     randomMeal.append(random.choice(RightMealList))

    todayMeal = []
    if len(testMealList1) > 0:
        randomMeal[0] = testMealList1[0]
    if len(testMealList2) > 0:
        randomMeal[1] = testMealList2[0]
    else:
        for i in range(len(mealKindList)):
            if mealKindList[i] == 0:
                randomMeal[1][0][i] = 13
                randomMeal[1][1][i] = '콩국수'
                randomMeal[1][2][i] = -1
                break
        

    for i in randomMeal:
        resultList = []
        for j in range(todayEatNum):
            if i[2][j] == -1:
                store_id_select_query = select_str('foods',
                                        ['store_id'],
                                        {'food_id': str(i[0][j])})
                cur.execute(store_id_select_query)
                store_id = cur.fetchall()[0][0]

                img_link = get_image('images2/'+str(store_id)+'/'+i[1][j]+'/0.png')
                resultList.append({'id': i[0][j], 'recommend_name' : i[1][j],
                    'image' : 'data:image/png;base64,'+img_link , 'store_id': store_id})

                # img_link = get_image('images2/'+str(store_id)+'/'+i[1][j]+'/0.png')
                # img_link = 'data:image/png;base64,'+img_link

                # img_link = os.path.abspath('./images2/' + str(store_id) + '/' + str(i[1][j]) + '/' + '0.png')
                # resultList.append({'id': i[0][j], 'recommend_name' : i[1][j],
                #     'image' : img_link , 'store_id': store_id})

                # get_image('images2/'+str(store_id)+'/'+i[1][j]+'/0.png')
                # with open('images2/'+str(store_id)+'/'+i[1][j]+'/0.png', 'rb') as f:
                #     base64_img = base64.b64encode(f.read())
                #     resultList.append({'id': i[0][j], 'recommend_name' : i[1][j],
                #        'image' : str(base64_img) })
            #     resultList.append({'id': i[0][j], 'recommend_name' : i[1][j],
            #             'image' : str(img_link) })
            else:
                resultList.append({'id': i[0][j], 'recommend_name' : i[1][j],
                        'image' : i[2][j] , 'store_id' : str(-1) })
        todayMeal.append(resultList)
    
    if -1 in FailCheckes or 1 in FailCheckes:
        mealInfoStr = "저번주 "
        nutrientIndex = ['탄수화물', '단백질', '지방']
        for i in range(len(FailCheckes)):
            if FailCheckes[i] == 1:
                mealInfoStr += nutrientIndex[i]+'을 많이 '
            elif FailCheckes[i] == -1:
                mealInfoStr += nutrientIndex[i]+'을 적게 '
        mealInfoStr += '섭취한 당신에 추천드리는 '

        for i in range(len(FailCheckes)):
            if FailCheckes[i] == 1:
                mealInfoStr += '저'+nutrientIndex[i]+' '
            elif FailCheckes[i] == -1:
                mealInfoStr += '고'+nutrientIndex[i]+' '
        mealInfoStr += '식단'
        print(mealInfoStr)
        result = {"recommendMeals" : todayMeal , "mealInfo": mealInfoStr}
    else:
        result = {"recommendMeals" : todayMeal}
    
    # print(todayMeal)
    return jsonify(result);


@app.route('/send', methods=['POST'])
def sendTest():
    tb._SYMBOLIC_SCOPE.value = True
    sendForm2 = request.form
    sendFile2 = request.files
    # print(sendForm2)
    # print('=============================')
    # print(sendFile2)
    # print(sendFile2[0])
    # print(len(sendFile2))
    # for i in range(5):
    #     print(sendFile2['스파게티'][i])s
    menuInfo = sendForm2['menus'].split(',')
    storeId = sendForm2['store_id']
    if not os.path.isdir('./images2/'+str(storeId)):
        os.mkdir('./images2/'+str(storeId))
    print(menuInfo)
    for i in menuInfo:
        if i:
            index = 0
            print(i)
            if not os.path.isdir('./images2/'+str(storeId)+'/'+str(i)):
                os.mkdir('./images2/'+str(storeId)+'/'+str(i))
            for j in sendFile2:
                if j.find(i) != -1:
                    if not os.path.isfile('./images2/'+str(storeId)+'/'+str(i)+'/'+str(index) +'.png'):
                        im = Image.open(sendFile2[j])
                        im.save('./images2/'+str(storeId)+'/'+str(i)+'/'+str(index) +'.png')
                        print('./images2/'+str(storeId)+'/'+str(i)+'/'+str(index) +'.png')
                    index += 1
                
    def do_work(storeId):
        tb._SYMBOLIC_SCOPE.value = True
        build.increament_image(str(storeId))
        food.make_model(str(storeId))

    thread = threading.Thread(target=do_work, kwargs={'storeId': storeId})
    thread.start()

    return 'test ok'

@app.route('/store/<store_id>', methods=['GET'])
def getStore(store_id):
    storeId = store_id
    cur.execute('select * from stores where store_id = '+ str(storeId))
    storeInfo = cur.fetchall()
    cur.execute('select food_name from foods where store_id = '+ str(storeId))
    menuInfo = [ i[0] for i in cur.fetchall() ]
    resultList = []
    # store_gps_latitude 
    # store_gps_longitude 
    # gps_l = str(storeInfo[0][2]).split(',')[0]
    # gps_r = str(storeInfo[0][2]).split(',')[1]
    resultList.append({'recommend_address' : storeInfo[0][4], 'store_name': storeInfo[0][1],
            'recommend_menu': menuInfo, 'gps_l': storeInfo[0][2], 'gps_r': storeInfo[0][3]})
    result = {"recommend" : resultList }

    return jsonify(result)

@app.route('/recipe/<recipe_id>', methods=['GET'])
def getRecipe2(recipe_id):
    # fw = open('recipe.csv', 'r')
    # recipeRdr = csv.reader(fw)

    # recipeList = [ line for line in recipeRdr ]
    recipeId = recipe_id
    cur.execute('select * from recipes where recipe_id = ' + str(recipeId))
    recipeList = cur.fetchall()
    print(recipeList)


    cur.execute('select nutrient_calorie, nutrient_carbohydrate, nutrient_protein, nutrient_fat, nutrient_salt from nutrients where recipe_id =' + str(recipeId))
    recipeNutrientList = cur.fetchall()
    
    resList = {}
    recipes = {}
    for i in recipeList:
        recipe = []
        for j in range(9, len(i)):
            if str(i[j]).find('http') == -1 or str(i[j]) == '\\r':
                if str(i[j]).strip():
                    recipe.append(i[j])
        recipes = recipe
        resList = [i[1],i[6],i[8],i[4]]

    resultList = []

    resultList.append({'foot_id': recipe_id, 'foot_name' : resList[0], 'foot_img' : resList[1],
    'foot_weight': resList[3], 'foot_calorimetry':  recipeNutrientList[0][0], 'foot_Carbohydrate' : recipeNutrientList[0][1],
        'foot_protein' : recipeNutrientList[0][2], 'foot_local': recipeNutrientList[0][3], 'foot_natrium': recipeNutrientList[0][4],
        'material' : resList[2], 'foot_recipe': recipes}, )

    result = {"footrecommend" : resultList }

    return jsonify(result)

def insert_str(table, colums, values):
    query_colum = '('
    for i in range(len(colums)):
        if i + 1 == len(colums):
            query_colum += colums[i]+')'
        else:
            query_colum += colums[i]+','
    value_query = ' values('
    for i in range(len(values)):
        if i + 1 == len(colums):
            value_query += '\''+values[i]+'\');'
        else:
            value_query += '\''+values[i]+'\','
    query = 'insert into ' + table + query_colum + value_query
    return query

def select_str(table, colums, values):
    query_colum = ' '
    for i in range(len(colums)):
        if i + 1 == len(colums):
            query_colum += colums[i]
        else:
            query_colum += colums[i]+','
    value_query = ''
    for key, value in values.items():
        value_query += key + ' = \'' + value + '\';'
        if len(values.items()) > 1:
            value_query += ' and '
    query = 'select' + query_colum + ' from ' + table + ' where ' + value_query
    return query


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
