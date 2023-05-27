import React, { useEffect, useRef, useState } from 'react';
import { View, Text, Image, Button } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { MobileNet } from "@tensorflow-models/mobilenet"
import * as ImagePicker from 'expo-image-picker';
import { decode, encode } from 'base-64';

if (!global.btoa) {
  global.btoa = encode;
}

if (!global.atob) {
  global.atob = decode;
}

const App = () => {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState('');
  const [imagePick, setImagePick] = useState("");
  const [error, setError] = useState("");
  const [model, setModel] = useState({} as MobileNet);
  const [ready, setReady] = useState(true);

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let resultImage = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      base64: true,
      quality: 1,
    });

    setResult("");
    setReady(false);
    if (!resultImage.canceled) {
      setImagePick(resultImage.assets[0].uri);
      setError("");
      // Convert the base64 encoded image data to a Uint8Array
      const base64Data = resultImage.assets[0].base64;
      const binaryString = atob(base64Data);
      // Create a Uint8Array from the binary string
      const uint8Array = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        uint8Array[i] = binaryString.charCodeAt(i);
      }
      await process(uint8Array);
    }

  };

  const load = async () => {
    try {
      // Load mobilenet.
      await tf.ready();
      await mobilenet.load().catch(error => console.log(error, "test"))
        .then(model => {
          if (model) {
            setModel(model);
            setIsTfReady(true);
          }
        });
    }
    catch (err) {
      if (err) {
        setError(err.toString() + " error loading model");
      }
    }
  }

  const process = async (uri: Uint8Array) => {
    try {
      // Start inference and show result.      

      // const response = await FileSystem.readAsStringAsync(uri).catch(err => {
      //   if (err) {
      //     setError(err.toString() + " reading file prob");
      //   }
      // });
      // // const imageDataArrayBuffer = await response.arrayBuffer();
      // const buffer = Buffer.from(response, 'base64');
      // console.log(buffer)
      // const imageData = new Uint8Array(buffer);

      const imageTensor = decodeJpeg(uri);

      if (model) {
        const prediction = await model.classify(imageTensor);
        if (prediction && prediction.length > 0) {
          setResult(
            `${prediction[0].className} (${prediction[0].probability.toFixed(3)})`
          );
        }
      }
      setReady(true);
    } catch (err) {
      if (err) {
        setError(err.toString() + " processing problem");
        setReady(true);
      }
    }
  }

  useEffect(() => {
    load();
  }, [])

  return (
    <View
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Button title="Pick an image from camera roll" disabled={!isTfReady || !ready} onPress={pickImage} />
      {imagePick && <Image source={{ uri: imagePick }} style={{ width: 400, height: 400 }} />}
      {!isTfReady && <Text>Loading TFJS model...</Text>}
      {isTfReady && result === '' && <Text>Classifying...</Text>}
      {ready && <Text>{result}</Text>}
      {error != "" && <Text>{error}</Text>}
    </View>
  );
};

export default App;
