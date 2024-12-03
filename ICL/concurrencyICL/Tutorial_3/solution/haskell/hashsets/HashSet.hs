module HashSet
  ( HashSet
  , delete
  , deleteAll
  , hashSet
  , insert
  , insertAll
  , numBuckets
  , size
  , toList
  ) where

import Data.Array
import Data.Hashable
import qualified Data.List

-- A hash set is an integer-indexed array of buckets, and the size of the hash
-- set is also stored, for easy access
data HashSet a =
  HashSet (Array Int [a]) Int

-- Returns an empty hash set with the given number of buckets
hashSet :: Int -> HashSet a
hashSet bucketCount =
  HashSet (array (0, bucketCount - 1) [(i, []) | i <- [0 .. bucketCount - 1]]) 0

-- Helper to return the index of the bucket associated with a given element
bucketIndex :: (Eq a, Hashable a) => HashSet a -> a -> Int
bucketIndex hs x = hash x `mod` numBuckets hs

-- Helper to determine whether the hash set could do with resizing
policy :: HashSet a -> Bool
policy hs = size hs `div` numBuckets hs > 4

-- Returns hash set with element inserted (if not already present)
insert :: (Eq a, Hashable a) => a -> HashSet a -> HashSet a
insert x (HashSet buckets size) =
  let index = bucketIndex (HashSet buckets size) x
   in if x `elem` (buckets ! index)
      -- Nothing to do - return the hash set unchanged
        then HashSet buckets size
      -- The updated hash set is like the original but the element added to the
      -- relevant bucket
        else let updated =
                   HashSet
                     (buckets // [(index, x : (buckets ! index))])
                     (size + 1)
              in if policy updated
              -- The updated hash set is large enough that we need to resize it
                   then resize updated
                   else updated

-- Conventient way to insert lots of elements
insertAll :: (Eq a, Hashable a) => [a] -> HashSet a -> HashSet a
insertAll xs hs = foldr HashSet.insert hs xs

-- Returns hash set with element deleted (if already present)
delete :: (Eq a, Hashable a) => a -> HashSet a -> HashSet a
delete x (HashSet buckets size) =
  let index = bucketIndex (HashSet buckets size) x
   in if x `elem` (buckets ! index)
      -- The element was present, so return a hash set with a smaller size
      -- whose relevant bucket is updated
        then HashSet
               (buckets // [(index, Data.List.delete x (buckets ! index))])
               (size - 1)
      -- Nothing to do
        else HashSet buckets size

-- Conventient way to delete lots of elements
deleteAll :: (Eq a, Hashable a) => [a] -> HashSet a -> HashSet a
deleteAll xs hs = foldr HashSet.delete hs xs

-- How many buckets does the hash set have?
numBuckets :: HashSet a -> Int
numBuckets (HashSet buckets size) = snd (bounds buckets) + 1

-- How many elements are in the hash set?
size :: HashSet a -> Int
size (HashSet buckets size) = size

-- Get all elements of the hash set as a big list
toList :: HashSet a -> [a]
toList (HashSet buckets size) =
  concatMap (buckets !) [0 .. (snd (bounds buckets))]

-- Redistribute the hash sets contents among twice as many buckets
resize :: (Eq a, Hashable a) => HashSet a -> HashSet a
resize hs = foldr HashSet.insert (hashSet (2 * numBuckets hs)) (toList hs)

-- String representation of hash set
instance (Show a) => Show (HashSet a) where
  show hs = show (toList hs)
