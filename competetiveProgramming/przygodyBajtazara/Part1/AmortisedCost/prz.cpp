//https://szkopul.edu.pl/problemset/problem/bobLSP2Wo3SQjakifBIQhXlq/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

// Debug
const bool DEBUG = false;

// Data
int n, k, l, r;
vector<int> houses, leftBandit, rightBandit; 

// Tracking sequences
int meetColor;
map<int, vector<int>> colorToIndexes;
vector<int> leftBanditIdxIndexes, rightBanditIdxIndexes;

// Tracking same color houses
int leftMostIdx, rightMostIdx, countLeftAndRight = 0;
map<int, int> leftColorToCount, rightColorToCount;

bool cinBandit(vector<int> &bandit, vector<int>&indexes, int size) {
    for (int i = 0; i < size; i++) {
        int house;
        cin >> house;
        bandit.push_back(house);
        indexes.push_back(0);
    }
    
    return true;
}

bool cinData() {
    cin >> n >> k;

    // Cin houses
    for (int i = 0; i < n; i++) {
        int house;
        cin >> house;
        houses.push_back(house);

        colorToIndexes[houses[i]].push_back(i);
    }

    // cin bandits
    cin >> l >> r;
    bool success = cinBandit(leftBandit, leftBanditIdxIndexes, l);
    if (!success) return false;
    success = cinBandit(rightBandit, rightBanditIdxIndexes, r);
    if (!success) return false;

    meetColor = rightBandit[rightBandit.size() - 1];
    rightBandit.pop_back(); leftBandit.pop_back();

    // intitiate indexes
    leftMostIdx = colorToIndexes[leftBandit[0]][0];
    rightMostIdx = colorToIndexes[rightBandit[0]][0];
    
    // Initiate counts
    for (int i = 0; i < leftMostIdx; i++) {
        leftColorToCount[houses[i]]++;
    }

    for (int i = rightMostIdx + 1; i < houses.size(); i++) {
        rightColorToCount[houses[i]]++;
        if (leftColorToCount[houses[i]] > 0 && rightColorToCount[houses[i]] == 1) {
            countLeftAndRight++;
        }
    }
    return true;
}

void updateCounts(int newLeftMostIdx, int newRightMostIdx) {
    // Update left
    for (int i = leftMostIdx; i <= newLeftMostIdx; i++) {
        leftColorToCount[houses[i]]++;
        if (leftColorToCount[houses[i]] == 1 && rightColorToCount[houses[i]] > 0) {
            countLeftAndRight++;
        }
    }
    leftMostIdx = newLeftMostIdx;
    // Update right
    // ERROR HERE !!!!!!!!!!!!!!!!!!!!!!!
    for (int i = rightMostIdx; i <= newRightMostIdx; i++) {
        rightColorToCount[houses[i]]++;
        if (rightColorToCount[houses[i]] == 1 && leftColorToCount[houses[i]] > 0) {
            countLeftAndRight--;
        }
    }
}

bool haveSameHouses() {
    return countLeftAndRight > 0;
}

bool updateSequence(int middleIdx) {
    // Increment left bandit
    int nextIdx = middleIdx;
    for (int i = leftBandit.size() - 1; i >= 0; i--) {
        int& curIdxIdx = leftBanditIdxIndexes[i];
        int prevIdxIdx = curIdxIdx;
        vector<int>& indexes = colorToIndexes[leftBandit[i]];

        while (curIdxIdx < indexes.size() && indexes[curIdxIdx] <= nextIdx) {
            curIdxIdx++;
        }

        if (curIdxIdx + 1 == indexes.size() && indexes[curIdxIdx - 1] <= nextIdx) {
            return false;
        } else {
            curIdxIdx--;
        }

        if (DEBUG) {
             cout << "---------\n";
            cout << "leftBandit[i]: " << leftBandit[i] << "\n";
            cout << "prevIdxIdx: " << prevIdxIdx << "\n";
            cout << "curIdxIdx: " << curIdxIdx << "\n";
            cout << "---------\n";
        }
       
        // No change so no need to update
        if (curIdxIdx == prevIdxIdx) {
            break;
        }
    }

    // Increment right bandit
    nextIdx = middleIdx;
    for (int i = leftBandit.size() - 1; i >= 0; i--) {
        int& curIdxIdx = rightBanditIdxIndexes[i];
        int prevIdxIdx = curIdxIdx;
        vector<int>& indexes = colorToIndexes[rightBandit[i]];

        while (curIdxIdx < indexes.size() && indexes[curIdxIdx] <= nextIdx) {
            curIdxIdx++;
        }

        if (curIdxIdx = indexes.size()) {
            return false;
        }

        if (DEBUG) {
            cout << "---------\n";
            cout << "rightBandit[i]: " << rightBandit[i] << "\n";
            cout << "prevIdxIdx: " << prevIdxIdx << "\n";
            cout << "curIdxIdx: " << curIdxIdx << "\n";
            cout << "---------\n";
        }

        // No change so no need to update
        if (curIdxIdx == prevIdxIdx) {
            break;
        }
    }
    return true;
}

int main() {
    bool b = cinData();
    if (!b) return 0;

    vector<int> ans;
    for (auto meet: colorToIndexes[meetColor]) {
        bool b = updateSequence(meet);
        
        if (DEBUG) {
            cout << "xxxxxxxxxxxxxxxxx\n";
            cout << "meet: " << meet << "\n";
            cout << "b: " << b << "\n";
        }

        if (!b) break;
        updateCounts(
            colorToIndexes[leftBandit[0]][leftBanditIdxIndexes[0]],
            colorToIndexes[rightBandit[0]][rightBanditIdxIndexes[0]]
        );

        if (haveSameHouses()) {
            ans.push_back(meet);
        }

        if (DEBUG) {
            cout << "leftMostIdx: " << leftMostIdx << "\n";
            cout << "rightMostIdx: " << rightMostIdx << "\n";
            cout << "countLeftAndRight: " << countLeftAndRight << "\n";
            cout << "xxxxxxxxxxxxxxxxx\n";
        }
    }

    cout << ans.size() << "\n";
    for (int i = 0; i < ans.size(); i++) {
        cout << ans[i] + 1 << " ";
    }
    cout << "\n";
}