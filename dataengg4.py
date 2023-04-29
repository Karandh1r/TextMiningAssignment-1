

# s = "aaaaaaabbbbccc"
# dict = {}
# list = []
# for i in range(len(s)):
#     if s[i] not in dict:
#         dict[s[i]] = 1
#     else:
#         dict[s[i]] += 1
# for key,value in dict.items():
#     list.append(tuple((key,value)))
# print(list)    

# def getyear(year,quater,add_quater):
#     if add_quater > 0:
#         years_add = add_quater // 4
#         quaters_add = add_quater % 4
#         year = year + years_add
#         quater = quaters_add + quater
#         if quater > 4:
#             years_add = quater // 4
#             year = year + years_add
#             quater = quater % 4
#     if add_quater < 0:
#         years_sub = abs(add_quater) // 4
#         quaters_sub = abs(add_quater) % 4
#         year = year - years_sub
#         quater = quater - quaters_sub
#         if quater < 0:
#             year = year - 1
#             quater = 4 - abs(quater)
#     return tuple((year,quater))        
# print(getyear(2020,4,8),end=" ")
# print(getyear(2020,4,11),end= " ")
# print(getyear(2021,4,-11),end= " ")
# print(getyear(2021,1,-11),end=" ")



  
# /*CREATE TABLE ORDER_PRODUCT(
#     OrderId int,
#     ProductId int 
# );*/

# /*INSERT INTO ORDER_PRODUCT VALUES (1,123);
# INSERT INTO ORDER_PRODUCT VALUES (2,123);
# INSERT INTO ORDER_PRODUCT VALUES (2,456);
# INSERT INTO ORDER_PRODUCT VALUES (2,789);
# INSERT INTO ORDER_PRODUCT VALUES (3,456);*/



# select p.productId,p.ProductName,c.name from Product_category as c 
# inner join  
# Product as p on c.product_category_Id = p.product_category_id
# inner join 
# (select ProductId,count(ProductId) from ORDER_PRODUCT group by ProductId
# having count(ProductId) in (select max(c) from (select *, count(ProductId) as c from ORDER_PRODUCT 
# group by ProductId))) as max_pro on p.productId = max_pro.ProductId order by p.productId; 




select c.customerId from Customer as c
inner join
Purchase_Order as on p_order on c.customerId = p_order.customerId
inner join 
Order_products as on order_pro on p_order.orderId = order_pro.orderId
inner join
Product as product on order_pro.productId = product.productId
inner join 
product_category as pro_cat on product.product_Id = pro_cat.product_category_id