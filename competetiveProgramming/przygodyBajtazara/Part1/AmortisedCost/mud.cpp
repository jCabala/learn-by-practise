// https://szkopul.edu.pl/problemset/problem/x9RNkgNzmCv2JLyWwqZo4wMG/site/?key=statement
#include <bits/stdc++.h>
using namespace std;
#define ll long long

const int MAXL = 351;
const bool DEBUG = false;

int l, m, ansX, ansY; // l := number of line; m := citizens in the capital
vector<ll> linesDst[MAXL], linesPop[MAXL]; // Distances and populations
ll cumPop[MAXL], cumDist[MAXL], cumCapCost[MAXL], sumCumCapCost, sumCumPop, minCost;

void printDebugHelpers() {
    if (!DEBUG) return;
    cout << "cumDist: ";
    for (int i = 0; i < l; i++) cout << cumDist[i] << " ";
    cout << "\n";

    cout << "cumCapCost: ";
    for (int i = 0; i < l; i++) cout << cumCapCost[i] << " ";
    cout << "\n";

    cout << "cumPop: ";
    for (int i = 0; i < l; i++) cout << cumPop[i] << " ";
    cout << "\n";
}

void printDebugCost(ll cost, int i, int j) {
    if (!DEBUG) return;
    cout << "i, j, cost: " << i << " " << j << " " << cost << "\n";
}

int main() {
    cin >> l >> m;

    int k;
    ll dst, pop;
    
    for(int i = 0; i < l; i++) {
        cin >> k;

        for (int j = 0; j < k; j++) {
           cin >> dst >> pop;
           linesDst[i].push_back(dst);
           linesPop[i].push_back(pop); 

           // Calculating a cumulative populations and costs to drive to the capiol
           cumDist[i] += dst;
           cumCapCost[i] += cumDist[i] * pop;
           cumPop[i] += pop;
        }
        sumCumCapCost += cumCapCost[i];
        sumCumPop += cumPop[i];
    }

    printDebugHelpers();

    // Calculating cost for the capitol
    minCost = sumCumCapCost;
    ansX = ansY = 0; 
    printDebugCost(minCost, 0, 0);

    // Considering each line and a city at each line
    for(int i = 0; i < l; i++) {
        ll restCosts = sumCumCapCost - cumCapCost[i];
        ll restCumPop = m + sumCumPop - cumPop[i];
        ll curLineLostCost = 0;
        ll remCurPop = cumPop[i];

        if (DEBUG) {
            cout << "LINE: " << i << "\n";
            cout << "restCosts, restCumPop: " << restCosts << " " << restCumPop << "\n";
        }

        for (int j = 0; j < linesDst[i].size(); j++) {
            restCosts += linesDst[i][j] * restCumPop;
            curLineLostCost += linesDst[i][j] * remCurPop;
            remCurPop -= linesPop[i][j];
            restCumPop += linesPop[i][j];

            ll cost = restCosts + cumCapCost[i] - curLineLostCost;

            if (DEBUG) {
                cout << "restCosts, restCumPop, curLineLostCost: " << restCosts << " " << restCumPop
                     << " " << curLineLostCost << "\n";
            }
            printDebugCost(cost, i + 1, j + 1);

            if (cost < minCost) {
                minCost = cost;
                ansX = i + 1;
                ansY = j + 1;
            }
        }
    }
    
    cout << minCost << "\n";
    cout << ansX << " " << ansY << "\n";
}