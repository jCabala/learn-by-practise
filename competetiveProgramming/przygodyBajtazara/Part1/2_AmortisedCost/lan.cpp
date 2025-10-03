//https://szkopul.edu.pl/problemset/problem/MAWN1VdLdXO29VvrVYuYxQyw/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

int solve(vector<int> lengths, vector<int> colors, vector<int> chain) {
    int n = chain.size(), m = lengths.size();
    map<int, int> color_to_cur, color_to_len;
    int good_chain_len = 0;
    for (int i = 0; i < m; i++) {
        color_to_len[colors[i]] = lengths[i];
        good_chain_len += lengths[i];
    }
    int num_good = 0, left = 0, ans = 0;
    for (int right = 0; right < n; right++) {
        color_to_cur[chain[right]]++;
        if (color_to_cur[chain[right]] == color_to_len[chain[right]]) {
            num_good++;
        }
        if (right - left + 1 == good_chain_len) {
            if (num_good == m) {
               ans++;
            }
            color_to_cur[chain[left]]--;
            if (color_to_cur[chain[left]] == color_to_len[chain[left]] - 1) {
                num_good--;
            }
            left++;
        }
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    vector<int> lengths, colors, chain;
    int n, m;
    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int l;
        cin >> l;
        lengths.push_back(l);
    }
    for (int i = 0; i < m; i++) {
        int c;
        cin >> c;
        colors.push_back(c);
    }
    for (int i = 0; i < n; i++) {
        int c;
        cin >> c;
        chain.push_back(c);
    }
    cout << solve(lengths, colors, chain) << "\n";
}