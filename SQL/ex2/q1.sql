SELECT name, character
FROM playsIn NATURAL JOIN actors
WHERE character LIKE 'George%'
ORDER BY name ASC, character ASC;
