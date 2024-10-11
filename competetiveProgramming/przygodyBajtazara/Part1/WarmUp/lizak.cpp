// https://szkopul.edu.pl/problemset/problem/AWhdD7i4V7mupdKWVtpgfGSM/site/?key=statement
#include <bits/stdc++.h>

using namespace std;

int m, n, maxPrice;
vector<int> lizak; // T == price 2, W == price 1
bool possible[2000001], reversed;
pair<int, int> ans[2000001];

void updateAt(int price, int beg, int end);

void findPossible() {
    //cout << "I START";
    int firstOneIdx = 0, curPrice = maxPrice;
    while(lizak[firstOneIdx] == 2) {
        curPrice -= lizak[firstOneIdx++];
        updateAt(curPrice, firstOneIdx, n - 1);
    }
    //cout << "I RUN";
    for (int i = n - 1; i >= firstOneIdx; i--) {
        updateAt(curPrice, firstOneIdx, i);
        updateAt(curPrice - 1, firstOneIdx + 1, i);
        curPrice -= lizak[i];
    }

    //cout << "I RUN TOO";
}

void updateAt(int price, int beg, int end) {
    possible[price] = true;
    reversed 
        ? ans[price] = {beg, end} 
        : ans[price] = {n - 1 - end, n - 1 - beg};
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0);
    cin >> n >> m;
    for (int i = 0; i < n; i++) {
        char c;
        cin >> c;
        int price = c == 'T' ? 2 : 1;
        lizak.push_back(price);
        maxPrice += price;
    }

    updateAt(maxPrice, 0, n - 1);
    findPossible();
    reverse(lizak.begin(), lizak.end());
    reversed = true;
    findPossible();
    //cout << "\n";
    for (int i = 0; i < m; i++) {
        int price;
        cin >> price;
        if (possible[price]) {
            auto pr = ans[price];
            cout << ans[price].first + 1 << " " << ans[price].second + 1 << "\n";
        } else {
            cout << "NIE\n";
        }
    }

    for (int i = 0; i <= maxPrice; i++) {
        if (possible[i]) {
            cout << i << " ";
        }
    }
}
