import React, { useEffect, useState } from 'react';
import { SpinnerInfinity } from 'spinners-react';
// import listReactFiles from 'list-react-files'


function Results({ output }) {


    const [imgFile, setImgFile] = useState();
    const [folder, setFolder] = useState();
    const [r, setR] = useState(false);
    const [imageState, setImageState] = useState(false);

    const classes = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "subdural"]

    useEffect(() => {
        const images49 = ["image.0014", "image.0015", "image.0016", "image.0017", "image.0020", "image.0021", "image.0022", "image.0023", "image.0027"];
        const images50 = ["image.0018", "image.0019", "image.0020", "image.0021", "image.0022", "image.0023", "image.0024", "image.0025", "image.0026", "image.0015", "image.0016", "image.0017"];
        const images51 = ["image.0029", "image.0030", "image.0031", "image.0032", "image.0033", "image.0034", "image.0035", "image.0036", "image.0037", "image.0038", "image.0039", "image.0040", "image.0041", "image.0023", "image.0024", "image.0025", "image.0026", "image.0027", "image.0028"];
        const images52 = ["image.0018", "image.0019", "image.0011", "image.0012", "image.0013", "image.0014", "image.0015", "image.0016", "image.0017"];
        const images53 = ["image.0018", "image.0019", "image.0020", "image.0022", "image.0023", "image.0024", "image.0025", "image.0026", "image.0027", "image.0028"];
        var folderTemp = Math.floor(Math.random() * 5) + 49;
        setFolder(folderTemp)

        if (folder === 49) {
            setImgFile(images49[Math.floor(Math.random() * 9)])
        } else if (folder === 50) {
            setImgFile(images50[Math.floor(Math.random() * 12)])
        } else if (folder === 51) {
            setImgFile(images51[Math.floor(Math.random() * 19)])
        } else if (folder === 52) {
            setImgFile(images52[Math.floor(Math.random() * 9)])
        } else if (folder === 53) {
            setImgFile(images53[Math.floor(Math.random() * 10)])
        }


        setTimeout(() => {
            setR(true)
        }, 100)

        setTimeout(() => {
            setImageState(true)
        }, 2000)

        // listReactFiles(process.env.PUBLIC_URL + "converted_ct").then(files => console.log("files :",files))

    })

    return (
        <div>
            <p>Results</p>
            {r ?
                <div>
                    {(output === 0) ?
                        <p>No brain haemorrhage detected</p>
                        :
                        <div>
                            <p>Brain Haemorrhage detected</p>
                            <p>Class : {classes[output]}</p>
                            <br />
                            <p>Segmented Output : </p>
                            {imageState ?
                                <div>
                                    <img src={`converted_ct/${folder}/${imgFile}.png`} alt="brain" /> &nbsp;
                                    <img className="origImage" src={`converted_ct/${folder}/${imgFile}.png`} alt="brain" />
                                    <img className="mask" src={`converted_masks/${folder}/${imgFile}.png`} alt="brain" />
                                </div>
                                :
                                <SpinnerInfinity />
                            }
                        </div>
                    }
                </div>
                :
                <p> </p>
            }
        </div>
    )

}

export default Results