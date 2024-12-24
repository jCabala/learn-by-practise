// https://szkopul.edu.pl/problemset/problem/bobLSP2Wo3SQjakifBIQhXlq/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

const int MAX_N = 1000001;
const int MAX_K = 1000001;

int N, K, meetHouseColor;
int houses[MAX_N], nextPtr[MAX_N], sequence[MAX_N];
int banditPosition[MAX_K], banditColor[MAX_N];
int rangeEndingForIdx[2][MAX_N], result[2][MAX_K];

void setupBandit(int banditSize, int banditNum) {
    // Creating a linked list of pointers to the next house of the same color
    fill(banditPosition, banditPosition+K, N);
    for (int j = N-1; j >= 0; j--) {
        nextPtr[j] = banditPosition[houses[j]];
        banditPosition[houses[j]] = j;
    }

    // Reading the colors of the houses for the ith rober
    for (int j = 1; j <= banditSize; j++) {
        cin >> banditColor[j];
        banditColor[j]--;
    }
    meetHouseColor = banditColor[banditSize]; // Used later in main

    // Creating the initial sequence for the first color
    // Iterate for each color and setup "left most" sequence
    int curPos = 0;
    for (int j = 1; j <= banditSize; j++) {
        while (banditPosition[banditColor[j]] < curPos)
            banditPosition[banditColor[j]] = nextPtr[banditPosition[banditColor[j]]];
        curPos = banditPosition[banditColor[j]];
        sequence[j] = curPos;
    }

    for (int j = 0; j < N; j++) {
        int curPos = 0;
        sequence[0] = j;

        // We stop if we didn't overshoot
        while (curPos <= banditSize && sequence[curPos] >= sequence[curPos+1]) {
            while (sequence[curPos] >= sequence[curPos+1] && sequence[curPos+1] < N)
                sequence[curPos+1] = nextPtr[sequence[curPos+1]];
            curPos++;
        }

        rangeEndingForIdx[banditNum][j] = sequence[banditSize]; // The last house of the sequence for the color 
    }   
} 

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> N >> K;

    for (int i = 0; i < N; i++) {
        cin >> houses[i];
        houses[i]--;
    }

    int size1, size2;
    cin >> size1 >> size2;
    setupBandit(size1, 0);
    fill(result[0], result[0]+N, N);
    for (int i = 0; i < N; i++)
        result[0][houses[i]] = min(result[0][houses[i]], rangeEndingForIdx[0][i]);

    reverse(houses, houses+N);
    setupBandit(size2, 1);
    for (int i = 0; i < N; i++)
        result[1][houses[i]] = max(result[1][houses[i]], N-1-rangeEndingForIdx[1][i]);
    reverse(houses, houses+N);

    // Finding the biggest range of houses where we can search
    pair<int, int> finalRes = {N, 0};
    vector<int> solution;
    for (int i = 0; i < K; i++)
        if (result[0][i] <= result[1][i]) { // If we have a valid range
            finalRes.first = min(finalRes.first, result[0][i]);
            finalRes.second = max(finalRes.second, result[1][i]);
        }

    for (int j = finalRes.first; j <= finalRes.second; j++)
        if (houses[j] == meetHouseColor)
            solution.push_back(j);


    cout << solution.size() << "\n";
    for (auto u : solution)
        cout << u+1 << " ";
    cout << "\n";

    return 0;
}
