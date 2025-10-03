// https://szkopul.edu.pl/problemset/problem/UU2Uj-barjiONnRxd9aEVoDj/site/?key=statement
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int n, m;
int grey[250001];

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        ++grey[a];
        ++grey[b];
    }

    ll  numDifferent = 0;
    for (int i = 1; i <= n; i++) {
        ll numGrey = static_cast<ll>(grey[i]);
        ll numBlack = static_cast<ll>(n - grey[i] - 1);
        numDifferent += numGrey * numBlack;
    }

    numDifferent /= 2;

    ll allTriangles = n * (n - 1) * (n - 2) / 6;
    ll numSame = allTriangles - numDifferent;

    cout << numSame << "\n";
}