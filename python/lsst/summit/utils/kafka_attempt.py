import requests
from kafka import KafkaConsumer
import json


def create_kafka_postISRmedian_topics():
    """This function creates the kafka topics for postISR pixel counts.
    Note: I do not understand if you're supposed to only do this once
    or do it every time a new cluster is made, or something else.
    Thus I don't really know how this function should be written or if it
    should be a function at all."""

    # Note: I'm not sure if you want this on usdf or the summit
    # or the option of doing both
    sasquatch_rest_proxy_urls = [
        "https://summit-lsp.lsst.codes/sasquatch-rest-proxy",
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy",
    ]

    headers = {"content-type": "application/json"}

    # make a list of the topics you want to create
    # I have no idea what reasonable partition counts or replication
    # factors are, so I just copied from the tutorial
    # Not sure of the correct topic name either
    all_topic_configs = [
        {
            "topic_name": "lsst.dm.latiss.postIsrPixelMedian",
            "partitions_count": 1,
            "replication_factor": 3,
        },
        {
            "topic_name": "lsst.dm.comcam.postIsrPixelMedian",
            "partitions_count": 1,
            "replication_factor": 3,
        },
    ]

    for sasquatch_url in sasquatch_rest_proxy_urls:
        # get cluster id
        r = requests.get(f"{sasquatch_url}/v3/clusters", headers=headers)

        cluster_id = r.json()["data"][0]["cluster_id"]

        headers = {"content-type": "application/json"}

        # create your kafka topics
        for topic_config in all_topic_configs:
            response = requests.post(
                f"{sasquatch_url}/v3/clusters/{cluster_id}/topics",
                json=topic_config,
                headers=headers,
            )

        print(response.text)  # yes I know this is terrible and I should use a logger


def post_to_sasquatch_latiss_isr(timestamp, obsid, postIsrPixelMedian):
    """I think this function posts to sasquatch"""

    # not sure again if this will be summit or usdf
    url = (
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.dm.latiss.postIsrPixelMedian"
    )

    payload = {
        "value_schema": '{"namespace": "lsst.dm.latiss", "type": "record", \
            "name": "postIsrPixelMedian", "fields": \
                [{"name": "timestamp", "type": "long"}, \
                    {"name": "obsid", "type": "integer"}, \
                        {"name": "instrument", "type": "string", "default": "LATISS"}, \
                            {"name": "postIsrPixelMedian","type": "float"}]}',
        "records": [
            {
                "value": {
                    "timestamp": timestamp,
                    "obsid": obsid,
                    "instrument": "LATISS",
                    "postIsrPixelMedian": postIsrPixelMedian,
                }
            }
        ],
    }
    headers = {
        "Content-Type": "application/vnd.kafka.avro.v2+json",
        "Accept": "application/vnd.kafka.v2+json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)


def post_to_sasquatch_comcam_isr(
    timestamp,
    obsid,
    postIsrPixelMedian,
    postIsrPixelMedianMedian,
    postIsrPixelMedianMean,
    postIsrPixelMedianMax,
):
    """I think this function posts to sasquatch"""

    # not sure again if this will be summit or usdf
    url = (
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.dm.latiss.postIsrPixelMedian"
    )

    payload = {
        "value_schema": '{"namespace": "lsst.dm.comcam", "type": "record", \
            "name": "postIsrPixelMedian", "fields": [{"name": "timestamp", \
                "type": "long"}, {"name": "obsid", "type": "integer"}, \
                    {"name": "instrument", "type": "string", "default": "ComCam"}, \
                        {"name": "postIsrPixelMedian","type": "float"},\
                            {"name": "postIsrPixelMedianMedian","type": "float"}, \
                                {"name": "postIsrPixelMedianMean","type": "float"}, \
                                    {"name": "postIsrPixelMedianMax","type": "float"}]}',
        "records": [
            {
                "value": {
                    "timestamp": timestamp,
                    "obsid": obsid,
                    "instrument": "ComCam",
                    "postIsrPixelMedian": postIsrPixelMedian,
                    "postIsrPixelMedianMedian": postIsrPixelMedianMedian,
                    "postIsrPixelMedianMean": postIsrPixelMedianMean,
                    "postIsrPixelMedianMax": postIsrPixelMedianMax,
                }
            }
        ],
    }
    headers = {
        "Content-Type": "application/vnd.kafka.avro.v2+json",
        "Accept": "application/vnd.kafka.v2+json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)


""" Making the listener"""


def listen_to_kafka(topic, key, group_id, broker):
    """
    topic is Topic name, e.g. 'lsst.dm.latiss.postIsrPixelMedian'
    key is key e.g. postIsrPixelMedian
    """

    # Kafka consumer configuration
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=broker,  # Kafka broker(s) - e.g. ['localhost:9092']
        group_id=group_id,  # Consumer group ID (for load balancing) - no clue
        auto_offset_reset="latest",  # Start reading from the latest message (ignore past messages)
        enable_auto_commit=True,  # Automatically commit the offsets
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),  # Deserialize JSON messages
    )

    print("Waiting for Kafka messages...")

    # Infinite loop to keep the consumer active
    for message in consumer:
        # This block is executed as soon as a message is received
        print(f"New message received: {message.value}")

        # Extract a value from the message
        value = message.value.get(key)
        print(f"Extracted value: {value}")
        return value

        # Perform your desired processing on the message


"""
# Kafka user - not sure where this lives

apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaUser
metadata:
  name: toyapp
  labels:
    # The name of the Strimzi ``Kafka`` resource, probably "sasquatch"
    strimzi.io/cluster: sasquatch

  authentication:
    # This should always be "tls"
    type: tls

  authorization:
    type: simple
    acls:

      # If your app consumes messages, this gives permission to consume as
      # part of any consumer group that starts with the named prefix.
      - resource:
          type: group
          name: "lsst.dm"
          patternType: prefix
        operations:
          - "Read"
        host: "*"

# Kafka access - not sure where this lives either

apiVersion: access.strimzi.io/v1alpha1
kind: KafkaAccess
metadata:
  name: toyapp-kafka
spec:
  kafka:
    # The name and namespace of the Strimzi ``Kafka`` resource, probably
    # "sasquatch", but not sure here...
    name: sasquatch
    namespace: sasquatch
    # This should always be "tls"
    listener: tls
  user:
    kind: KafkaUser
    apiGroup: kafka.strimzi.io
    # This is the name of the ``KafkaUser`` that you created
    name: toyapp
    # This is the namespace of the ``KafkaUser``, NOT your app's namespace,
    # probably "sasquatch"
    namespace: sasquatch


# providing credentials... this lives in the ap's container...

apiVersion: apps/v1
kind: Deployment
metadata:
 ...
  name: toyapp
  namespace: toyapp
spec:
  ...
  template:
    ...
    spec:
      containers:
      - env:
        - name: KAFKA_SECURITY_PROTOCOL
            secretKeyRef:
              key: securityProtocol
              name: myapp-kafka
        - name: KAFKA_BOOTSTRAP_SERVERS
          valueFrom:
            secretKeyRef:
              key: bootstrapServers
              name: myapp-kafka
        - name: KAFKA_CLUSTER_CA_PATH
          value: /etc/kafkacluster/ca.crt
        - name: KAFKA_CLIENT_CERT_PATH
          value: /etc/kafkauser/user.crt
        - name: KAFKA_CLIENT_KEY_PATH
          value: /etc/kafkauser/user.key

        ...

        volumeMounts:
        - mountPath: /etc/kafkacluster/ca.crt
          name: kafka
          subPath: ssl.truststore.crt
        - mountPath: /etc/kafkauser/user.crt
          name: kafka
          subPath: ssl.keystore.crt
        - mountPath: /etc/kafkauser/user.key
          name: kafka
          subPath: ssl.keystore.key

      ...

      volumes:
      - name: kafka
        secret:
          defaultMode: 420
          # The ``metadata.name`` value from the ``KafkaAccess`` resource in
          # your app's namespace
          secretName: toyapp-kafka
"""
