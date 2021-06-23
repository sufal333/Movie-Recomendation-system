// Databricks notebook source
import breeze.linalg._
import breeze.numerics._

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD


val NUM_USERS = 943
val NUM_MOVIES = 1682
val NUM_RATINGS = 100000

val ratingsRDD = sc.textFile("dbfs:/FileStore/shared_uploads/tamaraychowdhury@gmail.com/u.data")
.map(line => {
    val cols = line.split('\t')
    val userId = cols(0).toLong
    val movieId = cols(1).toLong + 50000 // to avoid mixups with userId
    val rating = cols(2).toDouble
    (userId, movieId, rating)
  })


  

// COMMAND ----------

val movieId2Title = sc.textFile("dbfs:/FileStore/shared_uploads/tamaraychowdhury@gmail.com/u.item")

.map(line => {
    val cols = line.split('|')
    val movieId = cols(0).toLong + 50000
    val title = cols(1)
    movieId -> title
  })
.collectAsMap

// COMMAND ----------

val movieId2Title_b = sc.broadcast(movieId2Title)

// COMMAND ----------

val userVerticesRDD: RDD[(VertexId, String)] = ratingsRDD.map(triple => (triple._1, "NA"))
val movieVerticesRDD: RDD[(VertexId, String)] = ratingsRDD.map(triple => (triple._2, movieId2Title_b.value(triple._2)))
val verticesRDD = userVerticesRDD.union(movieVerticesRDD)


// COMMAND ----------

val relationshipsRDD: RDD[Edge[Double]] = ratingsRDD.map(triple => Edge(triple._1, triple._2, triple._3))

val graph = Graph(verticesRDD, relationshipsRDD)

print("%d vertices, %d edges\n".format(graph.vertices.count, graph.edges.count))
assert(graph.edges.count == NUM_RATINGS)


// COMMAND ----------

// inputs
val sourceUserId = 21L
val p = 100 // number of users to look at
val q = 10  // number of movies to recommend

// COMMAND ----------

val moviesRatedByUser = graph.edges.filter(e => e.srcId == sourceUserId)
  .map(e => (e.dstId, e.attr))
  .collect
  .toMap

// COMMAND ----------

val moviesRatedByUser_b = sc.broadcast(moviesRatedByUser)
println("# movies rated by user: %d".format(moviesRatedByUser.size))


// COMMAND ----------

val usersWhoRatedMovies = graph.aggregateMessages[List[Long]](
    triplet => { // map function
      // consider only movies that user u has rated
      if (moviesRatedByUser_b.value.contains(triplet.dstId)) {
        // take user and send to the movie to be aggregated
        triplet.sendToDst(List(triplet.srcId))
      }
    },
    // reduce userIds into single list
    (a, b) => (a ++ b)
  )                                                 // (movieId, List[userId])
  .flatMap(rec => {
    val movieId = rec._1
    val userIds = rec._2
    userIds.map(userId => (userId, 1))              // (userId, 1)
  })
  .reduceByKey((a, b) => a + b)                     // (userId, n)
  .map(rec => rec._1)                               // unique List(userId)
  .collect
  .toSet

// COMMAND ----------

val usersWhoRatedMovies_b = sc.broadcast(usersWhoRatedMovies)
println("# unique users: %d".format(usersWhoRatedMovies.size))

// COMMAND ----------

// step 3: find p users with most similar taste as u
def buildVector(elements: List[(Long, Double)]): DenseVector[Double] = {
  val vec = DenseVector.zeros[Double](NUM_MOVIES)
  elements.foreach(e => {
    val vecIdx = (e._1 - 50001).toInt
    val vecVal = e._2
    vec(vecIdx) = vecVal
  })
  vec
}

def cosineSimilarity(vec1: DenseVector[Double], vec2: DenseVector[Double]): Double = {
  (vec1 dot vec2) / (norm(vec1) * norm(vec2))
}


// COMMAND ----------

val userVectorsRDD: RDD[(VertexId, DenseVector[Double])] = graph
  .aggregateMessages[List[(Long, Double)]](
    triplet => { // map function
      // consider only users that rated movies M
      if (usersWhoRatedMovies_b.value.contains(triplet.srcId)) {
        // send to each user the target movieId and rating
        triplet.sendToSrc(List((triplet.dstId, triplet.attr)))
      }
    },
    // reduce to a single list
    (a, b) => (a ++ b)
  )                                        // (userId, List[(movieId, rating)])
  .mapValues(elements => buildVector(elements)) // (userId, ratingVector)


val sourceVec = userVectorsRDD.filter(rec => rec._1 == sourceUserId)
  .map(_._2)
  .collect
  .toList(0)
val sourceVec_b = sc.broadcast(sourceVec)

val similarUsersRDD = userVectorsRDD.filter(rec => rec._1 != sourceUserId)
  .map(rec => {
    val targetUserId = rec._1
    val targetVec = rec._2
    val cosim = cosineSimilarity(targetVec, sourceVec_b.value)
    (targetUserId, cosim)
  })
val similarUserSet = similarUsersRDD.takeOrdered(p)(Ordering[Double].reverse.on(rec => rec._2))
  .map(rec => rec._1)
  .toSet
val similarUserSet_b = sc.broadcast(similarUserSet)
println("# of similar users: %d".format(similarUserSet.size))


// COMMAND ----------

val candidateMovies = graph.aggregateMessages[List[Long]](
    triplet => { 
      if (similarUserSet_b.value.contains(triplet.srcId) &&
         !moviesRatedByUser_b.value.contains(triplet.dstId)) {
        // send message [movieId] back to user
        triplet.sendToSrc(List(triplet.dstId))
      }
    },
    // reduce function
    (a, b) => a ++ b
  )                                             // (userId, List(movieId))
  .flatMap(rec => {
    val userId = rec._1
    val movieIds = rec._2
    movieIds.map(movieId => (movieId, 1))       // (movieId, 1)
  })
  .reduceByKey((a, b) => a + b)                 // (movieId, count)
  .map(_._1)                                    // (movieId)
  .collect
  .toSet

// COMMAND ----------

val candidateMovies_b = sc.broadcast(candidateMovies)
println("# of candidate movies for recommendation: %d".format(candidateMovies.size))

// COMMAND ----------


val recommendedMoviesRDD: RDD[(VertexId, Double)] = graph
  .aggregateMessages[List[Double]](
    triplet => { // map function
      // limit search to movies rated by top p similar users
      if (candidateMovies_b.value.contains(triplet.dstId)) {
        // send ratings to movie nodes
        triplet.sendToDst(List(triplet.attr))
      }
    },
    // reduce ratings to single list per movie
    (a, b) => (a ++ b)
  )
  .mapValues(ratings => ratings.foldLeft(0D)(_ + _) / ratings.size)

val recommendedMovies = recommendedMoviesRDD.takeOrdered(q)(Ordering[Double].reverse.on(rec => rec._2))
println("#-recommended: %d".format(recommendedMovies.size))

print("---- recommended movies ----\n")
recommendedMovies.foreach(rec => {
  val movieId = rec._1.toLong
  val score = rec._2
  val title = movieId2Title(movieId)
  print("(%.3f) [%d] %s\n".format(score, movieId - 50000, title))
})
