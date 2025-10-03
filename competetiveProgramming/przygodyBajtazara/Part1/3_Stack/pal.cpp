#include <bits/stdc++.h>
using namespace std;

#define ll long long

// Solution:
// Greedy algorithm with precomputation of next cheaper station using a monotonic stack.
// Idea: buy just enough to reach the first strictly cheaper station (if one exists within tank range)
// otherwise fill up.

const int DEBUG = false;

struct Station {
    int price;
    int dist;  // distance to next station
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int capacity, n;
    ll totalDist = 0;
    cin >> capacity >> n;

    vector<Station> stations(n);
    for (int i = 0; i < n; i++) {
        cin >> stations[i].price >> stations[i].dist;
        totalDist += stations[i].dist;
    }

    ll cost = 0;
    ll fuel = 0;

    // Precompute prefix sums of distances
    vector<ll> prefix(n + 1, 0);
    for (int i = 0; i < n; i++)
        prefix[i + 1] = prefix[i] + stations[i].dist;

    // Precompute next cheaper station index using monotonic stack
    vector<int> nextCheaper(n, -1); // -1 mean no cheaper station
    stack<int> st;
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && stations[st.top()].price >= stations[i].price)
            st.pop();
        if (!st.empty()) nextCheaper[i] = st.top();
        st.push(i);
    }

    if (DEBUG) {
        cout << "nextCheaper: ";
        for (int i = 0; i < n; i++) cout << nextCheaper[i] << " ";
        cout << "\n";
    }

    for (int i = 0; i < n; i++) {
        long long distToNext = stations[i].dist;

        int nc = nextCheaper[i], newFuel = fuel, newCost = cost;
        if (nc != -1 && prefix[nc] - prefix[i] <= capacity) {
            // There's a cheaper station within reach
            if (DEBUG)
                cout << "At the station " << i << " we had a cheaper station within reach\n";

            ll need = prefix[nc] - prefix[i];
            if (fuel < need) {
                newCost += (need - fuel) * 1LL * stations[i].price;
                newFuel = need;
            }
        } else {
            // No cheaper within reach -> fill up tank (or just enough to get to the end if within reach)
            if (DEBUG)
                cout << "At the station " << i << " we had NO cheaper station within reach\n";

            ll remToEnd = prefix[n] - prefix[i];
            ll target = min<ll>(capacity, remToEnd);

            if (fuel < target) {
                newCost += (target - fuel) * 1LL * stations[i].price;
                newFuel = target;
            }
        }

        if (DEBUG)
            cout << "At the station " << i << " we bough " << (newFuel - fuel) << " fuel for " << (newCost - cost) << " cost\n";

        cost = newCost;
        fuel = newFuel;
        fuel -= distToNext;
    }

    cout << cost << "\n";
    return 0;
}
