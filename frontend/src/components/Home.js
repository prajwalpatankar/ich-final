import React, { useState } from 'react';
import axios from 'axios';
import Results from './results';
import { SpinnerInfinity } from 'spinners-react';



function Home() {

  const [dicomFile, setDicomFile] = useState();
  const [loadOutput, setLoadOutput] = useState(false)
  const [classificationOp, setClassificationOp] = useState();

  const handleImageChange = (e) => {
    setDicomFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    setLoadOutput(false)
    e.preventDefault();
    console.log(dicomFile);
    let form_data = new FormData();
    form_data.append('dicomFile', dicomFile, dicomFile.name);
    let url = 'http://127.0.0.1:8000/uploaddicom/';
    axios.post(url, form_data, {
      headers: {
        'content-type': 'multipart/form-data',
      }
    })
      .then(res => {
        console.log(res);




        let get_url = 'http://127.0.0.1:8000/model_output?id=' + res.data.id
        axios.get(get_url)
          .then(res => {
            console.log(res);
            setClassificationOp(res.data[0])
          })
          .catch(err => console.log(err))

        setTimeout(() => {
          setLoadOutput(true)
        }, 3000)
        window.scrollTo({
          top: 1840,
          left: 0,
          behavior: 'smooth'
        });

        // reroute to results
        // this.props.history.push("/results");
        // console.log("pushed")
      })
      .catch(err => console.log(err))
  };

  return (
    <div className="App">
      <div className="App-header">
        <br /><br /><br />
        <img src='b4.png' width={300}/>
        <h3>ICH Detection/Classification and Segmentation</h3>
        <form onSubmit={handleSubmit}>
          <p>
            <input type="file"
              id="model_pic"

              // accept="image/png, image/jpeg" 
              onChange={handleImageChange} required />
          </p>
          <input type="submit" />
        </form>
        


        <br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
        {loadOutput ?
          <Results output={classificationOp}/>
          :
          <div>
            <br /><br /><br /><br /><br /><br /><SpinnerInfinity />
          </div>
        }
        <br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
      </div>
    </div>
  );

}













// class Home extends Component {

//   state = {
//     dicomFile: null
//   };

//   handleImageChange = (e) => {
//     this.setState({
//       dicomFile: e.target.files[0]
//     })
//   };

//   handleSubmit = (e) => {
//     e.preventDefault();
//     console.log(this.state);
//     let form_data = new FormData();
//     form_data.append('dicomFile', this.state.dicomFile, this.state.dicomFile.name);
//     let url = 'http://127.0.0.1:8000/uploaddicom/';
//     axios.post(url, form_data, {
//       headers: {
//         'content-type': 'multipart/form-data',
//       }
//     })
//       .then(res => {
//         console.log(res);




//         let get_url = 'http://127.0.0.1:8000/model_output?id=' + res.data.id
//         axios.get(get_url)
//           .then(res => {
//             console.log(res);

//           })
//           .catch(err => console.log(err))


//         // reroute to results
//         // this.props.history.push("/results");
//         // console.log("pushed")
//       })
//       .catch(err => console.log(err))
//   };

//   render() {
//     return (
//       <div className="App">
//         <div className="App-header">
//           <br /><br /><br /><br /><br />
//           <form onSubmit={this.handleSubmit}>
//             <p>
//               <input type="file"
//                 id="model_pic"
//                 // accept="image/png, image/jpeg" 
//                 onChange={this.handleImageChange} required />
//             </p>
//             <input type="submit" />
//           </form>


//           <br /><br /><br /><br /><br /><br /><br />
//           <Results />
//         </div>
//       </div>
//     );
//   }
// }

export default Home;