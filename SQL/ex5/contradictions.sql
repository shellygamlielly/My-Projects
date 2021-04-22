SELECT DISTINCT B1.name,B1.author,B1.year 
FROM bestsellers B1, bestsellers B2
WHERE (b1.name = b2.name) and (b1.author <> b2.author or b1.price <> b2.price or b1.reviews <> b2.reviews or b1.user_rating <> b2.user_rating) 
ORDER BY b1.name, b1.year;