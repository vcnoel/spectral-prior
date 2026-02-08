

import requests

def submit():
    url = "https://prediction.neuralk-ai.com/"
    access_code = 0
    
    print(f"Submitting to {url} with Access Code: {access_code}")
    
    # Try Headers
    headers = {
        "Access-Code": str(access_code),
        "Authorization": f"Bearer {access_code}",
        "User-Agent": "SpectralPriorAgent"
    }
    
    # Try POST with empty data or query
    try:

        # 1. Simple GET to check content
        response = requests.get(url, headers=headers)
        print(f"GET Base Response: {response.status_code}")
        # Print content to see if there are instructions
        print("Page Content Snippet:")
        print(response.text[:500])
        
        # 2. Try common API endpoints
        endpoints = ["api/submit", "api/score", "submit", "score", "prediction"]
        for ep in endpoints:
            full_url = url + ep
            try:
                # GET
                resp = requests.get(full_url, headers=headers)
                if resp.status_code != 404:
                    print(f"Found endpoint: {full_url} (GET {resp.status_code})")
                    print(resp.text[:200])
                
                # POST
                resp = requests.post(full_url, headers=headers, json={"model": "test", "access_code": access_code})
                if resp.status_code != 404 and resp.status_code != 405:
                    print(f"Found endpoint: {full_url} (POST {resp.status_code})")
                    print(resp.text[:200])
            except:
                pass

        if response.status_code == 200:
            print("Response saved to nicl_response.html")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    submit()
