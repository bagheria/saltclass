import sltcls


train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
train_y = [0, 1, 1]
vocab = ['statistics', 'medicine', 'crime']

obj_new = sltcls.SLT(train_X, train_y, vocabulary=vocab, language='en')
obj_new.enrich('kmeans', 2)
obj_new.train()
obj_new.print_info()
prediction = obj_new.predict(data_file='first_test.txt')
print(obj_new.newdata)
print(prediction)


# stc_object = STClassifier(train_X, train_y, vocabulary=['statistics', 'medicine', 'crime'], language='en')
# stc_object.kmeans_enrich(num_clusters=2)
# stc_object.train(classifier='SVM')
# stc_object.print_info()
# prediction = stc_object.predict(data_file='first_test.txt')
# print(stc_object.newdata)
# print(prediction)


# object_from_file = stclassifier.STClassifier.from_data_dir(train_dir='D:/train/', language='en')
# object_from_file.print_info()
# print(object_from_file.vocabulary)
# object_from_file.kmeans_enrich(num_clusters=2)
# print(object_from_file.X)
# object_from_file.train(classifier='SVM', gamma=3)
# prediction = object_from_file.predict(data_file='first_test.txt')
# print(object_from_file.newdata)
# print(prediction)

# print(STClassifier.__init__.__doc__)
# print(help(STClassifier.__init__))

