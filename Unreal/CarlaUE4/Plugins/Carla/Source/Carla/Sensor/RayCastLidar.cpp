// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla.h"
#include "Carla/Sensor/RayCastLidar.h"
#include <EngineGlobals.h>
#include <Runtime/Engine/Classes/Engine/Engine.h>
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "DrawDebugHelpers.h"
#include "Engine/CollisionProfile.h"
#include "Runtime/Engine/Classes/Kismet/KismetMathLibrary.h"
#include "StaticMeshResources.h"

//Variables for storing the labels
FName actor_temp;
std::string actor_string;
std::vector<float> string_ascii;
std::vector<float> angles;
TArray<UStaticMeshComponent*> Components;


//exception counter
int i;

FActorDefinition ARayCastLidar::GetSensorDefinition()
{
	return UActorBlueprintFunctionLibrary::MakeLidarDefinition(TEXT("ray_cast"));
}

ARayCastLidar::ARayCastLidar(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryActorTick.bCanEverTick = true;

	auto MeshComp = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RootComponent"));
	MeshComp->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
	MeshComp->bHiddenInGame = true;
	MeshComp->CastShadow = false;
	MeshComp->PostPhysicsComponentTick.bCanEverTick = false;
	RootComponent = MeshComp;
}


void ARayCastLidar::Set(const FActorDescription &ActorDescription)
{
	Super::Set(ActorDescription);
	FLidarDescription LidarDescription;
	UActorBlueprintFunctionLibrary::SetLidar(ActorDescription, LidarDescription);
	Set(LidarDescription);
}

void ARayCastLidar::Set(const FLidarDescription &LidarDescription)
{
	Description = LidarDescription;
	LidarMeasurement = FLidarMeasurement(Description.Channels);
	CreateLasers();
}

void ARayCastLidar::CreateLasers()
{
	const auto NumberOfLasers = Description.Channels;
	check(NumberOfLasers > 0u);
	const float DeltaAngle = NumberOfLasers == 1u ? 0.f :
		(Description.UpperFovLimit - Description.LowerFovLimit) /
		static_cast<float>(NumberOfLasers - 1);
	LaserAngles.Empty(NumberOfLasers);
	for (auto i = 0u; i < NumberOfLasers; ++i)
	{
		const float VerticalAngle =
			Description.UpperFovLimit - static_cast<float>(i) * DeltaAngle;
		LaserAngles.Emplace(VerticalAngle);
	}
}

void ARayCastLidar::Tick(const float DeltaTime)
{
	Super::Tick(DeltaTime);

	ReadPoints(DeltaTime);

	auto DataStream = GetDataStream(*this);
	DataStream.Send(*this, LidarMeasurement, DataStream.PopBufferFromPool());
}

void ARayCastLidar::ReadPoints(const float DeltaTime)
{
	const uint32 ChannelCount = Description.Channels;
	const uint32 PointsToScanWithOneLaser =
		FMath::RoundHalfFromZero(
			Description.PointsPerSecond * DeltaTime / float(ChannelCount));
	//GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("F %d"), PointsToScanWithOneLaser));
	if (PointsToScanWithOneLaser <= 0)
	{
		UE_LOG(
			LogCarla,
			Warning,
			TEXT("%s: no points requested this frame, try increasing the number of points per second."),
			*GetName());
		return;
	}

	check(ChannelCount == LaserAngles.Num());

	const float CurrentHorizontalAngle = LidarMeasurement.GetHorizontalAngle();
	const float AngleDistanceOfTick = Description.RotationFrequency * 360.0f * DeltaTime;
	
	const float AngleDistanceOfLaserMeasure = AngleDistanceOfTick / PointsToScanWithOneLaser;

	LidarMeasurement.Reset(ChannelCount * PointsToScanWithOneLaser);
	auto AngularRes = 360.0f / (Description.PointsPerSecond / Description.RotationFrequency);
	for (auto Channel = 0u; Channel < ChannelCount; ++Channel)
	{
		for (float i = 0; i < 360; i += AngularRes)
		{
			
			FVector Point;
			std::vector<std::string> Label;


			const float Angle = CurrentHorizontalAngle + i;//AngleDistanceOfLaserMeasure * i;
			if (ShootLaser(Channel, Angle, Point))
			{
				LidarMeasurement.WritePoint(Channel, Point, actor_string, Angle);
			}
		}
	}
	const float HorizontalAngle = std::fmod(CurrentHorizontalAngle + AngleDistanceOfTick, 360.0f);
	LidarMeasurement.SetHorizontalAngle(HorizontalAngle);;
}

bool ARayCastLidar::ShootLaser(const uint32 Channel, const float HorizontalAngle, FVector &XYZ) const
{
	const float VerticalAngle = LaserAngles[Channel];

	FCollisionQueryParams TraceParams = FCollisionQueryParams(FName(TEXT("Laser_Trace")), true, this);
	TraceParams.bTraceComplex = true;
	TraceParams.bReturnPhysicalMaterial = false;

	FHitResult HitInfo(ForceInit);

	FVector LidarBodyLoc = GetActorLocation();
	FRotator LidarBodyRot = GetActorRotation();
	FRotator LaserRot(VerticalAngle, HorizontalAngle, 0);  // float InPitch, float InYaw, float InRoll
	FRotator ResultRot = UKismetMathLibrary::ComposeRotators(
		LaserRot,
		LidarBodyRot
	);
	const auto Range = Description.Range;
	FVector EndTrace = Range * UKismetMathLibrary::GetForwardVector(ResultRot) + LidarBodyLoc;

	//bool is_hit = GetObstacle(actor_, LidarBodyLoc, EndTrace, HitInfo, actor_, ECC_Visibility);

	GetWorld()->LineTraceSingleByChannel(
		HitInfo,
		LidarBodyLoc,
		EndTrace,
		ECC_MAX,
		TraceParams,
		FCollisionResponseParams::DefaultResponseParam
	);

	if (HitInfo.bBlockingHit)
	{

		//lidar return actor
		if (HitInfo.Actor.Get() != nullptr)
		{
			FString actor_Fstring = HitInfo.Actor->GetActorLabel();
			//Fname to Fstring and Fstring to std::string
			//FString actor_Fstring = actor_temp.ToString();
			actor_string = std::string(TCHAR_TO_UTF8(*actor_Fstring));
		}
		else
		{
			i++;
			//GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("F %d"), i));
			actor_string = "Fetch Failed";
		}

		if (Description.ShowDebugPoints)
		{
			DrawDebugPoint(
				GetWorld(),
				HitInfo.ImpactPoint,
				10,  //size
				FColor(255, 0, 255),
				false,  //persistent (never goes away)
				0.1  //point leaves a trail on moving object
			);
		}

		XYZ = LidarBodyLoc - HitInfo.ImpactPoint;
		XYZ = UKismetMathLibrary::RotateAngleAxis(
			XYZ,
			-LidarBodyRot.Yaw + 90,
			FVector(0, 0, 1)
		);

		return true;
	}
	else
	{
		return false;
	}
}

