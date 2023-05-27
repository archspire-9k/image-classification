import React, { useEffect, useRef, useState } from 'react';
import { View, Text, Image, Button } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

const App = () => {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState('');
  const [imagePick, setImagePick] = useState("");
  const [error, setError] = useState();

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let resultImage = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!resultImage.canceled) {
      setImagePick(resultImage.assets[0].uri);
      // console.log("canceled")
      load();
    }
  };

  const load = async () => {
    try {
      // Load mobilenet.
      await tf.ready();
      const model = await mobilenet.load().catch(error => console.log(error, "test"));
      setIsTfReady(true);

      // Start inference and show result.
      // const image = require('./basketball.jpg');
      // const imageAssetPath = Image.resolveAssetSource(image);
      // const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
      const response = await FileSystem.readAsStringAsync(imagePick, { encoding: FileSystem.EncodingType.Base64 });
      // const imageDataArrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(response, 'base64');
      const imageData = new Uint8Array(buffer);
      
      const imageTensor = decodeJpeg(imageData);
      if (model) {
        const prediction = await model.classify(imageTensor);
        if (prediction && prediction.length > 0) {
          setResult(
            `${prediction[0].className} (${prediction[0].probability.toFixed(3)})`
          );
        }
      }
    } catch (err) {
      setError(err)
    }
  };

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
      <Button title="Pick an image from camera roll" disabled={!isTfReady} onPress={pickImage} />
      {imagePick && <Image source={{ uri: imagePick }} style={{ width: 200, height: 200 }} />}
      {!isTfReady && <Text>Loading TFJS model...</Text>}
      {isTfReady && result === '' && <Text>Classifying...</Text>}
      {result !== '' && <Text>{result}</Text>}
      {error && <Text>{error}</Text>}
    </View>
  );
};

export default App;
