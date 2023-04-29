#include<iostream>
#include<cstring>
#include<vector>
using namespace std;
 int longestCommonSubsequenceHelper(string text1,string text2,int idx1,int idx2,vector<vector<int>> &dp){
        if(idx1 < 0 || idx2 < 0){
            return 0;
        }
        if(dp[idx1][idx2] != 0){
            return dp[idx1][idx2];
        } 
        if(text1[idx1] == text2[idx2]){
            dp[idx1][idx2] = 1 + longestCommonSubsequenceHelper(text1,text2,idx1-1,idx2-1,dp);
            return dp[idx1][idx2];
        }
        else{
            int firstfull = longestCommonSubsequenceHelper(text1,text2,idx1,idx2-1,dp);
            int secondfull = longestCommonSubsequenceHelper(text1,text2,idx1-1,idx2,dp);
            dp[idx1][idx2] = max(firstfull,secondfull);
            return dp[idx1][idx2];
        }
    }
int main(){
    string text1 = "acd";
    string text2 = "ced";
    vector<vector<int>> dp(text1.length(), vector<int> (text2.length(), 0));
    int idx1 = text1.length() - 1;
    int idx2 = text2.length() - 1;
    cout<<longestCommonSubsequenceHelper(text1,text2,idx1,idx2,dp);
}