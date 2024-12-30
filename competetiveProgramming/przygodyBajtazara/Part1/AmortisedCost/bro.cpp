// https://szkopul.edu.pl/problemset/problem/Z1C91LB8rGYMxy6wRLBmbXba/site/?key=statement
#include <bits/stdc++.h>
using namespace std;
#define ll long long

const bool DEBUG = false;
const int MAXN = 1000001;
int n, l, r; // l/r := (split points) furthest city we deliver to the left/right
ll dmd[MAXN], dst[MAXN]; // Distance and demand
ll lDst, rDst, lDmd, rDmd, cost, minCost;

void debugPrint() {
    if (!DEBUG) return;
    cout << "l, r, cost: " << l << " " << r << " " << cost << "\n";
}

void adjustSplitPoints() {
    while (rDst + dst[r] < lDst) { // It is cheaper to deliver to l clockwise 
        cost = cost + dmd[l] * (dst[r] + rDst) - dmd[l] * lDst;
        rDst += dst[r];
        rDmd += dmd[l];
        lDst -= dst[l];
        lDmd -= dmd[l];      
        r = (r + 1) % n;
        l = (l + 1) % n;
    }
}

int main() {
    cin >> n;
    for (int i = 0; i < n; i++) cin >> dmd[i] >> dst[i];

    // Initial setup with brewery at 0 and delivering to everyone counterclockwise
    r = 0; l = 1;
    for(int i = n - 1; i > 0; i--) {
        lDst += dst[i];
        lDmd += dmd[i];
        cost += lDst * dmd[i];
    }
 
    adjustSplitPoints();
    minCost = cost;
    debugPrint();

    // Moving brewery to the right and finding optimal split points
    for (int i = 1; i < n; i++) {
        ll lastDst = dst[i - 1];
        cost = cost - rDmd * lastDst + (lDmd + dmd[i-1]) * lastDst;
        rDmd -= dmd[i];
        lDmd += dmd[i-1];
        rDst -= lastDst;
        lDst += lastDst;
        adjustSplitPoints();
        minCost = min(minCost, cost);
        debugPrint();
    }

    cout << minCost << "\n";
}