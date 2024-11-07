-- https://ninetynine.haskell.chungyc.org/Problems.html#g:2
myLast :: [a] -> Maybe a
myLast []     = Nothing
myLast [a]    = Just a
myLast (a:as) = myLast as

myLast2 :: [a] -> Maybe a
myLast2 [] = Nothing
myLast2 xs = Just ((head . reverse) xs)

elementAt :: [a] -> Int -> Maybe a
elementAt [] _      = Nothing
elementAt (x:xs) 1  = Just x
elementAt (x:xs) n  = elementAt xs (n - 1)

isPalindrome :: Eq a => [a] -> Bool
isPalindrome xs = (helper . (zip xs) . reverse) xs
    where
        helper :: Eq a =>[(a, a)] -> Bool
        helper []          = True
        helper ((a, b):xs) = if a == b then helper xs else False

isPalindrome2 :: Eq a => [a] -> Bool
isPalindrome2 xs = all (uncurry (==)) (zip xs (reverse xs))

-- >>> pack "aaaabccaadeeee"
-- ["aaaa","b","cc","aa","d","eeee"]
pack :: Eq a => [a] -> [[a]]
pack [] = []
pack xs = packHelper xs []
    where
        packHelper :: Eq a => [a] -> [[a]] -> [[a]]
        packHelper [] acc    = reverse acc
        packHelper (x:xs) [] = packHelper xs [[x]]
        packHelper (x:xs) acc
            | x == curChar = packHelper xs updatedAcc1
            | otherwise    = packHelper xs updatedAcc2
            where
                curChar = (head . head) acc
                updatedAcc1 = (curChar : (head acc)) : (tail acc) 
                updatedAcc2 = [x] : acc

pack2 :: Eq a => [a] -> [[a]]
pack2 [] = []
pack2 (x:xs) = (x : prefix) : pack2 suffix
  where
    (prefix, suffix) = span (== x) xs