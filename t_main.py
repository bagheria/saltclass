import saltclass
import numpy as np


train_X = np.array([[10, 0, 0], [0, 20, 0], [4, 13, 5]])
train_y = np.array([0, 1, 1])
vocab = ['statistics', 'medicine', 'crime']

object_from_df = saltclass.SALT(train_X, train_y, vocabulary=vocab, language='en')
X = np.array([[10, 12, 0], [14, 3, 52]])
object_from_df.enrich(method='kmeans', include_unlabeled=True, unlabeled_matrix=X)

object_from_df.enrich(method='kmeans', include_unlabeled=True, unlabeled_dir='D:/Data/unlabeled/')

object_from_file = saltclass.SALT.data_from_dir(train_dir='D:/data/train2/', language='en')
obj_new = object_from_file
obj_new.enrich()
obj_new.train(classifier='svm')
obj_new.print_info()
prediction = obj_new.predict(data_file='second_test.txt')
print(obj_new.vocabulary)
print(obj_new.newdata)
print([k for (k, v) in obj_new.vocabulary.items() if obj_new.newdata[0][v] != 0])
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

