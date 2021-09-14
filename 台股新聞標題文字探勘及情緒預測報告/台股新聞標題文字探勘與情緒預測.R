library(tidyverse)
library(ggplot2)
library(readxl)
library(tidytext)
library(jiebaR)
library(wordcloud)
library(tm)
library(SnowballC) # for text stemming
library(topicmodels)
library(wordcloud2)
require(e1071)
# 讀取資料＆切分訓練資料、測試資料  
df_train=read_xlsx("foreign_invest_train.xlsx", col_names = TRUE)
############## train dataset###################
text_train=df_train[,2]
text=text_train %>% deframe()

#################test dataset##################



#斷字處理（中文需斷詞，斷詞後輸出為1 vector）
stocks=read_xlsx("TWstock.xlsx",col_names = T)
stock=c(stocks[,2]) %>% deframe()
#seg<-worker(symbol = T,bylines = T)
# text_wb <- sapply(segment(text, seg), function(x){
#   paste(x, collapse = " ")})
# text_wb
# 新增新詞
fintext=c("權值股","庫藏股","成長股","成長股","台指期","台股","開高","走低","翻紅","翻黑","開低","走高","殖利率","摩台指","買超","賣超","每股盈餘","法說會","新台幣","科技業","新低","新高","法人","砍單","爆單","做多","做空","月營收")
new_words <- c(stock,fintext)
# 匯出新詞
writeLines(new_words, "new_words.txt")
# 設定停止詞
stop_words <- c("在","再","的","下","個","來","至","座","亦","與","或","日","月","年","週","天","這","元","檔","到","未","是","仍","對","成","後","點","逾","達","辦","受","會","被","前","後","自","億","兆","角","評","遭","有","估","幾檔","近","億元")
writeLines(stop_words, "stop_words.txt")


##去掉train英文及數字(兩種替換方式)
text<- str_replace_all(text,"[0-9a-zA-Z]+?" ,"")
text<-gsub("[[:punct:]]", "",text)




# 重新定義斷詞器，匯入停止詞、專有名詞：第二種方式
seg <- worker(user = "new_words.txt", stop_word = "stop_words.txt", bylines = T)
seg_words <- seg[text]


# 斷詞後貼回去
text_wb <- sapply(segment(text, seg), function(x){
  paste(x, collapse = " ")})


#####################計算詞彙在全文出現頻率#####################
seg_words1=seg_words %>% unlist() %>% as.vector()
# 計算詞彙頻率
txt_freq <- freq(seg_words1)
# 由大到小排列
txt_freq <- arrange(txt_freq, desc(freq))
# 檢查前5名
head(txt_freq)

###############文字雲###############


#一般wordcloud需要定義字體，不然會無法顯示中文
par(family=("Heiti TC Light")) #"STKaiti"
# 一般的文字雲 (pkg: wordcloud)
wordcloud(txt_freq$char, txt_freq$freq, min.freq = 3, random.order = F, ordered.colors = F, colors = c("gray20","gray80"))

#可以動的文字雲
#wordcloud2(filter(txt_freq, freq > 1), 
minSize = 2, fontFamily = "Microsoft YaHei", size = 1)
#調整文字雲形狀
#letterCloud(txt_freq,'R',size = 0.35)
##################################
par(family=("Heiti TC Light"))
#對高頻文字視覺話
txt_freq %>%
  filter(freq > 15) %>%
  mutate(char = reorder(char, freq)) %>% ggplot(aes(freq, char)) +
  geom_col() + labs(y = NULL)+ theme_grey(base_family = "STKaiti")



############################## tf-idf################################
text_df <- tibble(line = 1:288, text = text_wb)

#news=paste0("Doc",1:length(seg_words))
# x=cbind(topic=1,word=seg_words[[1]]) %>% as.data.frame()
# for(i in 2:length(seg_words)){
#   c=cbind(topic=i,word=seg_words[[i]])%>% as.data.frame()
#   x=rbind(x,c)
# }
#train
x=tibble(topic=1,word=seg_words[[1]])
for(i in 2:length(seg_words)){
  c=tibble(topic=i,word=seg_words[[i]])
  x=rbind(x,c)
}

#train
x %>% group_by(topic)%>%
  dplyr::count(, word) %>% 
    bind_tf_idf(term = word,document = topic,n)->tf.idf.table


#找出每一個new 前三重要的詞彙
# tf.idf.table %>% 
#    top_n(3,tf_idf) %>%  
#    ungroup() -> top3


#train
a=ungroup(tf.idf.table)
b=pivot_wider(a,names_from = topic, values_from = tf_idf)

c=b[,-(2:4)]
c[is.na(c)] <- 0
d=t(c)[-1,] %>% as.data.frame()
d=lapply(d, as.numeric)

y=df_train[,3] %>% as.data.frame()

dat <- data.frame(x=d, y=as.factor(y$label))
dat_train=dat[1:230,]
dat_test=dat[231:288,]
############################### SVM #######################
require(e1071)
####linear SVM ###########
svmfit <- svm(y ~ ., data=dat_train, kernel ="linear", cost=10, scale=FALSE)
set.seed(1)
tune.out <- tune(svm, y ~ ., data=dat_train ,kernel ="linear",
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary (tune.out)

bestmod <- tune.out$best.model 
summary(bestmod)

num.SV = sapply(X=0.001:100, 
                FUN=function(C) svm(y~., data_train, cost=C, epsilon =.1)$tot.nSV)

# 隨著C越大，support vectors數量越少，代表margin的範圍越窄，越接近hard-margin SVM，越容易overfitting
plot(x=1:1000, y=num.SV, xlab="C value", ylab="# of support vectors", pch=16, cex=.5, main="# of SVs in soft-margin SVM")




#predict
ypred <- predict(bestmod ,dat_test) 
table(predict=ypred, truth=dat_test$y)

######### non-linear SVM ############
#radial
svmfit_nl= svm(y ~ ., data = dat_train, kernel = "radial", gamma = 1, cost = 1) 
#plot(svmfit_nl, dat_train)
summary(svmfit_nl)
#cross validation
set.seed(1)
tune.out_nl= tune(svm, y ~ ., data = dat_train, kernel ="radial",
                ranges = list(cost=c(0.01, 0.1, 1, 10, 100), gamma = c(0.5,1,2,3)))
summary(tune.out_nl)
bestmod_nl <- tune.out_nl$best.model 
ypred_nl <- predict(bestmod_nl ,dat_test) 
table(true = dat_test$y, pred = ypred_nl)

#sigmoid
svmfit_sig= svm(y ~ ., data = dat_train, kernel = "sigmoid", gamma = 1, cost = 1) 
#plot(svmfit_nl, dat_train)
summary(svmfit_sig)
#cross validation
set.seed(1)
tune.out_sig= tune(svm, y ~ ., data = dat_train, kernel ="sigmoid",
                  ranges = list(cost=c(0.01, 0.1, 1, 10, 100), gamma = c(0.5,1,2,3)))
summary(tune.out_sig)
bestmod_sig <- tune.out_sig$best.model 
ypred_sig <- predict(bestmod_sig ,dat_test) 
table(true = dat_test$y, pred = ypred_sig)

# #polynomial 
# svmfit_pol= svm(y ~ ., data = dat_train, kernel = "polynomial", gamma = 1, cost = 1,degree=3) 
# #plot(svmfit_nl, dat_train)
# summary(svmfit_pol)
# #cross validation
# set.seed(1)
# tune.out_pol= tune(svm, y ~ ., data = dat_train, kernel ="polynomial",
#                    ranges = list(cost=c(0.01, 0.1, 1, 10, 100), gamma = c(0.5,1,2,3)))
# summary(tune.out_pol)
# bestmod_pol <- tune.out_pol$best.model 
# ypred_pol <- predict(bestmod_pol ,dat_test) 
# table(true = dat_test$y, pred = ypred_sig)
